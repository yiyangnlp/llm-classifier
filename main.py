from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from openai import OpenAI 

MODEL="gpt-4o-mini"
client = OpenAI()

app = FastAPI()

# Define the request model
class Example(BaseModel):
    text: str
    label: str

class ClassificationRequest(BaseModel):
    input_text: str
    labels: Dict[str, str]
    examples: Optional[List[Example]] = None


@app.post("/classify")
def classify_text(request: ClassificationRequest) -> Dict[str, str]:
    """
    Classify the input text based on provided labels and few-shot examples.

    Parameters:
    - request (ClassificationRequest): A model containing input text, labels, and examples.

    Returns:
    - Dict[str, str]: The predicted label.
    """
    prompt = generate_prompt(request.input_text, request.labels, request.examples)
    
    response = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )

    prediction = parse_response(response)
    return {"label": prediction}


def generate_prompt(input_text: str, labels: Dict[str, str], examples: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generates a prompt for the LLM based on input text, labels, and examples.

    Parameters:
    - input_text (str): The text to be classified.
    - labels (Dict[str, str]): Dictionary of label names and their descriptions.
    - examples (Optional[List[Dict[str, str]]]): A list of few-shot examples with 'text' and 'label'.

    Returns:
    - str: The generated prompt to be sent to the LLM.
    """
    label_descriptions = "\n".join([f"- {label}: {desc}" for label, desc in labels.items()])
    example_text = ""
    if examples:
        example_text += "Here are some example texts and their corresponding labels: \n\n"
        for ex in examples:
            example_text += f"Text: \"{ex.text}\"\nLabel: {ex.label}\n\n"
    prompt = (
        f"You are an assistant that classifies texts into the following categories:\n"
        f"{label_descriptions}\n\n"
        f"{example_text}"
        f"Please read the following text and provide the most appropriate label.\n"
        f"Text: \"{input_text}\"\n"
        f"Label: "
    )
    return prompt


def parse_response(response: Dict) -> str:
    """
    Parses the LLM's response to extract the predicted label.

    Parameters:
    - response (Dict): The response object from the OpenAI API.

    Returns:
    - str: The predicted label.
    """
    # Extract the text portion of the response
    text_response = response.choices[0].message.content.strip()

    # Alternatively, if the response contains just the label directly, we can return it as is
    return text_response
