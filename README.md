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

## Endpoint Discussion

## Option 1: Three Endpoints

- **/add_label** – Adds custom labels and descriptions.
- **/add_examples** – Adds few-shot examples.
- **/classify** – Classifies text using the provided labels and examples.

### Pros:
- **Separation of Concerns**: Each task (adding labels, adding examples, classifying) is handled by separate endpoints, making the code cleaner and easier to manage.
- **Modular Design**: Users can add labels and examples once and reuse them across multiple classification tasks. This reduces redundancy.
- **Reusability**: If you plan to classify multiple texts using the same labels and few-shot examples, this is more efficient since labels/examples don't need to be sent repeatedly.
- **Scalability**: The system can scale better because users can incrementally add labels and examples as needed.
- **Simplicity for Clients**: Clients (e.g., UI or other systems) can first configure labels and examples and then simply use the `/classify` endpoint for multiple classification tasks.

### Cons:
- **More API Calls**: Users have to make multiple API calls (one to add labels, another to add examples, and a final one to classify), which may increase complexity if the classification needs to be done quickly with minimal interaction.
- **State Management**: You need to manage state on the server to remember the labels and examples across requests (e.g., by user session, API key, or globally). This adds complexity in terms of caching, persistence, and cleanup.

---

## Option 2: Single /classify Endpoint

- **/classify** – Accepts labels, descriptions, examples, and text for classification all in a single request.

### Pros:
- **Simplified API**: Only one endpoint is needed, making it easy for users to interact with the API. All the data is passed in a single request, reducing the complexity for the client.
- **Stateless**: The system can remain stateless because all the information (labels, examples, and text) is passed with each request. There’s no need to store information on the server.
- **No Pre-Configuration Needed**: Users don’t have to worry about setting up labels and examples in advance—they provide everything in one go, which can be faster for one-off classifications.

### Cons:
- **Data Overhead**: If users need to classify multiple pieces of text using the same labels and examples, they’ll have to pass the labels and examples in every request, leading to redundant data and increased payload size.
- **Limited Flexibility**: If the same labels/examples are used repeatedly, clients will need to send the same data for each classification, which can be inefficient for repetitive tasks.
- **Complexity for Clients**: If users want to perform multiple classifications with different configurations (e.g., change labels or examples), they have to construct new requests each time, which can make client-side code more complex.


For simplicity and scalability, we chose the **one endpoint approach**.




## Experiment Report: Evaluating Text Classification Using Zero-Shot and 10-Shot Learning

---

### 1. Introduction

In this experiment, we evaluated the performance of a text classification model using the **Hugging Face** `datasets` library and a custom FastAPI-based classification endpoint. The goal of this experiment was to assess how well the model performs in both **zero-shot** (no examples provided) and **10-shot** (using 10 few-shot examples) settings. We conducted this evaluation on three datasets covering binary and multiclass classification tasks: **Yelp Polarity**, **AG News**, and **TREC**.

Simply run:

```bash
poetry shell
python eval.py
```

---

### 2. Datasets

The following datasets were used in the evaluation:

1. **Yelp Polarity (Binary Classification)**:
   - **Task**: Sentiment analysis, where reviews are classified as **Positive** or **Negative**.
   - **Labels**: `0` for Negative, `1` for Positive.
   - **Dataset Size**: 500,000 examples, with a subset of 200 examples used for testing.

2. **AG News (Multiclass Classification)**:
   - **Task**: Classifying news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.
   - **Labels**: `0` for World, `1` for Sports, `2` for Business, `3` for Sci/Tech.
   - **Dataset Size**: 120,000 examples, with a subset of 200 examples used for testing.

3. **TREC (Multiclass Classification)**:
   - **Task**: Question classification into six categories: **Description**, **Entity**, **Abbreviation**, **Human**, **Location**, and **Numeric**.
   - **Labels**: `0` for Description, `1` for Entity, `2` for Abbreviation, `3` for Human, `4` for Location, `5` for Numeric.
   - **Dataset Size**: 5,952 examples, with a subset of 200 examples used for testing.

For each dataset, the first **200 test examples** were used in the evaluation process. The first **10 train examples** were used as the few-shot examples.

---

### 3. Methods

**FastAPI `/classify` Endpoint**: 
- A custom text classification endpoint was implemented using FastAPI, which leveraged an LLM (here we used gpt-4o-mini) to classify text based on user-provided labels and examples.
- Two settings were evaluated for each dataset:
  1. **Zero-shot**: No examples were provided, and the model had to predict labels based solely on the input text and label descriptions.
  2. **10-shot**: The first 10 examples from the training set were provided as few-shot learning examples to guide the model's predictions.

**Evaluation Procedure**:
- For each dataset, the first 200 test examples were used.
- Accuracy was calculated by comparing the predicted labels from the `/classify` endpoint to the ground truth labels from the dataset.
- Accuracy scores for both **zero-shot** and **10-shot** settings were recorded.

---

### 4. Results

The table below summarizes the accuracy results across the three datasets for both **zero-shot** and **10-shot** settings:

| **Dataset**      | **Zero-Shot Accuracy** | **10-Shot Accuracy** |
|------------------|------------------------|----------------------|
| **Yelp Polarity** | 30.00%                 | 99.00%               |
| **AG News**       | 83.50%                 | 85.50%               |
| **TREC**          | 59.00%                 | 55.50%               |

---

### 5. Discussion

1. **Yelp Polarity**:
   - The **zero-shot accuracy** for Yelp Polarity was relatively low at 30%, due to the string label for the sentiments are `1` and `2` rather than `positive` and `negative`. However, in the **10-shot setting**, the model achieved near-perfect accuracy (99%), demonstrating that the model can leverage few-shot learning to drastically improve performance for binary classification tasks.

2. **AG News**:
   - For AG News, the model performed quite well in both settings, with **83.50% accuracy in zero-shot** and a modest improvement to **85.50% in the 10-shot** setting. The small gain from few-shot learning suggests that the model is relatively effective at classifying news topics without much additional context but benefits slightly from the provided examples.

3. **TREC**:
   - The **TREC dataset** results were somewhat surprising. The **zero-shot accuracy** was 59%, which is acceptable given the complexity of the classification task. However, the model's performance decreased slightly to **55.50% in the 10-shot** setting. This could be due to misaligned examples or the model's difficulty in generalizing from the provided few-shot examples, suggesting that further optimization in prompt engineering might be needed for complex question classification tasks.
