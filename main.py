import asyncio
import logging
import os
from typing import Any, Dict, List, Union

import openai
from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class LLMCompletionRequest(BaseModel):
    table_data: List[Dict[str, Any]] = Field(..., description="Array of rows from the spreadsheet.")
    target_column: str = Field(..., description="Name of the column whose value will be used in the LLM prompt.")
    prompt_template: str = Field(..., description="f-string like template for LLM prompt, e.g., 'Classify: {major_value}'.")
    new_column_name: str = Field(..., description="Name of the new column for LLM's completion.")

class LLMCompletionResponse(BaseModel):
    status: str = Field(..., description="'success' or 'error'.")
    data: List[Dict[str, Any]] = Field(..., description="Original table_data with the new column added.")
    message: str = Field(..., description="Descriptive message about the operation's outcome.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Spreadsheet LLM Completion Service",
    description="An API service to perform LLM-powered data completions on tabular data.",
    version="1.0.0"
)

# --- OpenAI Client Initialization ---
# Client will be initialized per request to ensure API key is fresh if changed,
# and to handle missing key error appropriately per request.

# --- LLM Call Logic with Retries ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError # For server-side errors from OpenAI (5xx)
    )),
    reraise=True # Reraise the exception if all retries fail
)
async def get_llm_completion_with_retry(prompt: str, client: openai.AsyncOpenAI) -> str:
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            return completion.choices[0].message.content.strip()
        else:
            logger.error(f"LLM returned empty or invalid response for prompt: {prompt}")
            return "LLM_ERROR_EMPTY_RESPONSE"
    except openai.AuthenticationError: # Handled specifically to fail the entire batch
        logger.error("OpenAI AuthenticationError in get_llm_completion_with_retry.")
        raise
    except (openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
        logger.warning(f"Retryable OpenAI API error: {type(e).__name__} - {e}. Retrying...")
        raise # Let tenacity handle retry
    except openai.OpenAIError as e: # Catch other OpenAI errors not covered by retry or auth
        logger.error(f"Non-retryable OpenAI error for prompt '{prompt}': {type(e).__name__} - {e}")
        return f"LLM_ERROR_API_OTHER: {type(e).__name__}"
    except Exception as e:
        logger.exception(f"Unexpected error calling OpenAI for prompt '{prompt}': {e}")
        return f"LLM_ERROR_UNEXPECTED: {type(e).__name__}"

# --- Row Processing Logic ---
async def process_row(
    row_data: Dict[str, Any],
    target_column_name: str,
    prompt_template_str: str,
    new_col_name: str,
    client: openai.AsyncOpenAI
) -> Dict[str, Any]:
    updated_row = dict(row_data) # Work with a copy
    try:
        if target_column_name not in updated_row:
            raise KeyError(f"Target column '{target_column_name}' not found in row data: {list(updated_row.keys())}")
        
        value_to_insert = updated_row[target_column_name]
        
        # Ensure the placeholder {major_value} exists in the template
        if "{major_value}" not in prompt_template_str:
             raise ValueError("Prompt template must contain '{major_value}' placeholder.")

        current_prompt = prompt_template_str.format(major_value=str(value_to_insert))
        
        llm_response_text = await get_llm_completion_with_retry(current_prompt, client)
        updated_row[new_col_name] = llm_response_text
    
    except KeyError as e: # Handles missing target_column
        logger.warning(f"KeyError during prompt construction for row ID {updated_row.get('id', '(no id)')}: {e}")
        updated_row[new_col_name] = f"Error: Input data issue ({e})"
    except ValueError as e: # Handles missing {major_value} in template
        logger.warning(f"ValueError during prompt construction for row ID {updated_row.get('id', '(no id)')}: {e}")
        updated_row[new_col_name] = f"Error: Prompt template issue ({e})"
    except openai.AuthenticationError: # Propagate for global failure
        raise
    except (openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e: # Retries exhausted
        logger.error(f"OpenAI API error after retries for row ID {updated_row.get('id', '(no id)')}: {type(e).__name__} - {e}")
        updated_row[new_col_name] = f"LLM_ERROR_API_RETRY_EXHAUSTED: {type(e).__name__}"
    except Exception as e: # Catch-all for other errors from get_llm_completion_with_retry or unexpected ones here
        logger.exception(f"Unexpected error processing row ID {updated_row.get('id', '(no id)')}: {e}")
        updated_row[new_col_name] = f"LLM_ERROR_PROCESSING_ROW: {type(e).__name__}"
    return updated_row

# --- API Endpoint ---
@app.post("/llm-complete", response_model=LLMCompletionResponse)
async def llm_complete_endpoint(request: LLMCompletionRequest, http_response: Response):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        http_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return LLMCompletionResponse(
            status="error", 
            data=request.table_data, 
            message="Server configuration error: Missing OpenAI API key."
        )
    
    try:
        # Use AsyncOpenAI for asyncio compatibility
        client = openai.AsyncOpenAI(api_key=openai_api_key)
    except Exception as e:
        logger.exception("Failed to initialize OpenAI client")
        http_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return LLMCompletionResponse(
            status="error", 
            data=request.table_data, 
            message=f"Server configuration error: Could not initialize OpenAI client - {type(e).__name__}."
        )

    tasks = [
        process_row(dict(r), request.target_column, request.prompt_template, request.new_column_name, client)
        for r in request.table_data
    ]
    
    # return_exceptions=True allows us to handle individual task failures, especially AuthenticationError
    results_from_gather: List[Union[Dict[str, Any], Exception]] = await asyncio.gather(*tasks, return_exceptions=True)

    final_processed_data: List[Dict[str, Any]] = []
    
    # Check for AuthenticationError first, as it should fail the entire request
    for i, res_item in enumerate(results_from_gather):
        if isinstance(res_item, openai.AuthenticationError):
            logger.error("OpenAI Authentication Failed during batch processing. Check API Key.")
            http_response.status_code = status.HTTP_401_UNAUTHORIZED # Or 500 as per spec "fail the entire request"
            return LLMCompletionResponse(
                status="error", 
                data=request.table_data, # Return original data
                message="OpenAI Authentication Failed. Please check your API key and ensure it's valid and has funds."
            )

    # If no AuthenticationError, process other results
    for i, res_item in enumerate(results_from_gather):
        original_row = request.table_data[i]
        if isinstance(res_item, Exception): # Should be errors already handled and converted by process_row, but this is a safeguard
            logger.error(f"Unhandled exception from process_row for original row index {i}: {res_item}")
            temp_row = dict(original_row)
            temp_row[request.new_column_name] = f"LLM_ERROR_UNHANDLED_IN_GATHER: {type(res_item).__name__}"
            final_processed_data.append(temp_row)
        elif isinstance(res_item, dict): # Successfully processed row (res_item is the updated dict)
            final_processed_data.append(res_item)
        else: # Should not happen if process_row always returns a dict or raises
            logger.error(f"Unexpected item type from process_row for original row index {i}: {type(res_item)}")
            temp_row = dict(original_row)
            temp_row[request.new_column_name] = "LLM_ERROR_UNKNOWN_PROCESS_ROW_OUTPUT"
            final_processed_data.append(temp_row)
            
    return LLMCompletionResponse(
        status="success", 
        data=final_processed_data, 
        message="LLM completions processed."
    )

# To run this application (after installing dependencies):
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
