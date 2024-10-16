# LLM Classifier

LLM Classifier is a FastAPI-based application that allows you to classify text using OpenAI's GPT-4. The API supports binary and multi-class classification and can be enhanced with few-shot examples to improve accuracy.

## Features

- Supports custom labels for text classification.
- Can handle both binary and multi-class classification tasks.
- Allows for few-shot learning by providing labeled examples to improve model predictions.

---

## Prerequisites

Before setting up the project, make sure you have the following installed:

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/docs/#installation)
- An [OpenAI API Key](https://beta.openai.com/signup/)

---

## Installation

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yiyangnlp/llm-classifier.git
   cd llm-classifier
   ```

2. **Install Dependencies**

   This project uses Poetry for dependency management. To install the project dependencies, run:

   ```bash
   poetry install
   ```

3. **Configure the OpenAI API Key**

   You need to set up your OpenAI API key as an environment variable:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```


## Running the Application

Once the dependencies are installed and the API key is configured, you can start the FastAPI server.


Use the following command to start the server:

```bash
poetry run uvicorn main:app --reload
```

By default, the server will run on `http://127.0.0.1:8000/`.


## Testing the `/classify` Endpoint

### 1. Using cURL

You can send a POST request to the `/classify` endpoint using cURL as follows:

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
-H "Content-Type: application/json" \
-d '{
    "input_text": "I love this product!",
    "labels": {
        "Positive": "The text expresses positive sentiment.",
        "Negative": "The text expresses negative sentiment."
    },
    "examples": [
        {"text": "This is the best day ever!", "label": "Positive"},
        {"text": "I hate everything about this.", "label": "Negative"}
    ]
}'
```

### 2. Using Python requests Library

You can also use Python to test the /classify endpoint:

```python
import requests

url = "http://127.0.0.1:8000/classify"

# Define the payload
payload = {
    "input_text": "I love this product!",
    "labels": {
        "Positive": "The text expresses positive sentiment.",
        "Negative": "The text expresses negative sentiment."
    },
    "examples": [
        {"text": "This is the best day ever!", "label": "Positive"},
        {"text": "I hate everything about this.", "label": "Negative"}
    ]
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print(response.json())
```