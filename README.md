# Spreadsheet UI Backend - LLM Completion Service

This project provides a FastAPI backend service for a spreadsheet-like UI. It allows the frontend to send tabular data along with a prompt template, and the backend uses an LLM (OpenAI's gpt-4o-mini) to generate new data for a specified column based on the template and existing column values.

## Features

-   Accepts tabular data, a target column, a prompt template, and a new column name.
-   Uses OpenAI's `gpt-4o-mini` model for completions.
-   Processes rows concurrently using `asyncio.gather`.
-   Implements retries with exponential backoff for transient OpenAI API errors.
-   Handles various errors gracefully, including API key issues, network problems, and rate limits.
-   Provides detailed per-row error reporting if an LLM call fails for a specific row (unless it's a critical error like authentication).

## API Endpoint

-   **Endpoint**: `/llm-complete`
-   **Method**: `POST`
-   **Request Body**: See `LLMCompletionRequest` in `main.py` or the example below.
-   **Response Body**: See `LLMCompletionResponse` in `main.py` or the example below.

## Project Setup

1.  **Clone the repository (if applicable) or ensure you have the project files.**

2.  **Create and activate a Python virtual environment:**

    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    # .\venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the project root directory by copying `.env.example`:

    ```bash
    cp .env.example .env
    ```

    Edit the `.env` file and replace `"your_openai_api_key_here"` with your actual OpenAI API key:

    ```
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

## Running the Backend Server

To run the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

-   `--reload`: Enables auto-reloading when code changes (for development).
-   `--host 0.0.0.0`: Makes the server accessible from your local network.
-   `--port 8000`: Specifies the port to run on.

The API will be available at `http://localhost:8000`.

## Testing the Endpoint

You can test the `/llm-complete` endpoint using `curl` or a Python script.

### Using `curl`

```bash
curl -X POST "http://localhost:8000/llm-complete" \
-H "Content-Type: application/json" \
-d '{
  "table_data": [
    {
      "id": 1,
      "First Name": "Alice",
      "Last Name": "Smith",
      "Major": "Computer Science",
      "Enrollment Date": "2022-09-01"
    },
    {
      "id": 2,
      "First Name": "Bob",
      "Last Name": "Johnson",
      "Major": "Literature",
      "Enrollment Date": "2023-01-15"
    },
    {
      "id": 3,
      "First Name": "Charlie",
      "Last Name": "Brown",
      "Major": "Electrical Engineering",
      "Enrollment Date": "2022-09-01"
    }
  ],
  "target_column": "Major",
  "prompt_template": "Classify the following university major as \'Engineer\' or \'Non-Engineer\': {major_value}. Respond with only \'Engineer\' or \'Non-Engineer\'.",
  "new_column_name": "EngineeringClassification"
}'
```

### Expected `curl` Output (example):

```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "First Name": "Alice",
      "Last Name": "Smith",
      "Major": "Computer Science",
      "Enrollment Date": "2022-09-01",
      "EngineeringClassification": "Engineer"
    },
    {
      "id": 2,
      "First Name": "Bob",
      "Last Name": "Johnson",
      "Major": "Literature",
      "Enrollment Date": "2023-01-15",
      "EngineeringClassification": "Non-Engineer"
    },
    {
      "id": 3,
      "First Name": "Charlie",
      "Last Name": "Brown",
      "Major": "Electrical Engineering",
      "Enrollment Date": "2022-09-01",
      "EngineeringClassification": "Engineer"
    }
  ],
  "message": "LLM completions processed."
}
```

*(Note: Actual LLM output may vary slightly.)*

### Using a Python `requests` script (Optional)

Create a Python script (e.g., `test_api.py`):

```python
import requests
import json

api_url = "http://localhost:8000/llm-complete"

request_payload = {
    "table_data": [
        {
            "id": 1,
            "First Name": "Alice",
            "Major": "Computer Science"
        },
        {
            "id": 2,
            "First Name": "Bob",
            "Major": "Philosophy"
        }
    ],
    "target_column": "Major",
    "prompt_template": "Briefly describe the field of {major_value} in one sentence.",
    "new_column_name": "MajorDescription"
}

try:
    response = requests.post(api_url, json=request_payload)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    print("Status Code:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), indent=2))
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    print("Response Content:", response.text)
except requests.exceptions.RequestException as req_err:
    print(f"Request exception occurred: {req_err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

Run the script: `python test_api.py`

## Error Handling

-   **Input Validation**: If the request body doesn't match the `LLMCompletionRequest` schema, FastAPI will return a `422 Unprocessable Entity` error.
-   **Missing `OPENAI_API_KEY`**: The server will return a `500 Internal Server Error` if the API key is not configured.
-   **OpenAI Authentication Error**: If the API key is invalid or has issues, the server will return a `401 Unauthorized` error for the entire request.
-   **Other OpenAI API Errors (Rate Limits, Connection Issues)**: These are retried up to 3 times. If they persist, the specific row's new column will be populated with an error message (e.g., `LLM_ERROR_API_RETRY_EXHAUSTED: RateLimitError`).
-   **Prompt Template Issues**: If `target_column` is not found in a row or `prompt_template` does not contain `{major_value}`, the specific row's new column will indicate an `Error: Input data issue (...)` or `Error: Prompt template issue (...)`.
-   **Partial Failures**: The service aims to process all rows. If an LLM call fails for a specific row after retries (and it's not an authentication error), that row's `new_column_name` will contain an error indicator (e.g., `LLM_ERROR_API_RETRY_EXHAUSTED`), while other rows are processed normally.

```
