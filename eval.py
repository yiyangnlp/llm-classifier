import requests
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Define the FastAPI /classify endpoint URL
url = "http://127.0.0.1:8000/classify"

def load_classification_dataset(dataset_name: str, split: str = "test"):
    """
    Load a classification dataset from Hugging Face.
    
    Parameters:
    - dataset_name (str): The name of the dataset (e.g., "yelp_polarity", "ag_news", "trec").
    - split (str): The dataset split to load (default is "test").
    
    Returns:
    - dataset: The loaded dataset split.
    - label_mapping (dict): A dictionary that maps label IDs to their string descriptions.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, split=split)
    
    # Extract the label names if available
    # Some datasets like TREC have 'coarse_label'
    if 'label' in dataset.features:
        label_names = dataset.features['label'].names
    else:
        label_names = dataset.features['coarse_label'].names
    
    # Map label indices to their names
    label_mapping = {i: label for i, label in enumerate(label_names)}
    
    return dataset, label_mapping


def create_payload(input_text: str, label_mapping: dict, examples=None):
    """
    Create the request payload for the /classify endpoint.
    
    Parameters:
    - input_text (str): The input text to classify.
    - label_mapping (dict): A dictionary mapping label IDs to their string descriptions.
    - examples (list): Optional few-shot examples to include in the request payload.
    
    Returns:
    - payload (dict): The formatted request payload.
    """
    # TODO: add better description of the labels
    labels = {label: f"Text is classified as {label}." for label in label_mapping.values()}
    
    payload = {
        "input_text": input_text,
        "labels": labels,
        "examples": examples
    }
    
    return payload


def evaluate_classification_model(dataset_name: str, few_shot: bool = False, max_examples: int = 200):
    """
    Evaluate the FastAPI /classify endpoint on a classification dataset.
    
    Parameters:
    - dataset_name (str): The name of the dataset (e.g., "yelp_polarity", "ag_news", "trec").
    - few_shot (bool): Whether to include few-shot examples in the payload (default is False).
    - max_examples (int): The maximum number of test examples to evaluate (default is 200).
    
    Returns:
    - accuracy (float): The classification accuracy on the test set.
    """
    # Load the test dataset and label mapping (use first 200 examples from the test set)
    dataset, label_mapping = load_classification_dataset(dataset_name, split="test[:200]")

    # Load the first 10 examples from the training set for few-shot learning
    if few_shot:
        train_dataset, _ = load_classification_dataset(dataset_name, split="train[:10]")
        few_shot_examples = [
            {"text": train_dataset[i]['text'], 
             "label": label_mapping[train_dataset[i]['label'] if 'label' in train_dataset[i] else train_dataset[i]['coarse_label']]}
            for i in range(10)
        ]
    else:
        few_shot_examples = None

    # Store true labels and predicted labels for evaluation
    true_labels = []
    predicted_labels = []

    # Iterate over the test set (limit to max_examples)
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break

        input_text = example['text']
        true_label = label_mapping[example['label'] if 'label' in example else example['coarse_label']]  # True label

        # Prepare the payload for the /classify endpoint
        payload = create_payload(input_text, label_mapping, few_shot_examples)

        # Send a POST request to the /classify endpoint
        response = requests.post(url, json=payload)
        prediction = response.json().get("label")

        # Append true and predicted labels for evaluation
        true_labels.append(true_label)
        predicted_labels.append(prediction)

    # Calculate and return the accuracy
    # print(true_labels)
    # print(predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy


# Evaluate on Yelp Polarity (binary classification) - Zero Shot and 10-Shot
yelp_accuracy_zero_shot = evaluate_classification_model("yelp_polarity", few_shot=False)
yelp_accuracy_few_shot = evaluate_classification_model("yelp_polarity", few_shot=True)
print(f"Yelp Polarity accuracy (Zero Shot): {yelp_accuracy_zero_shot * 100:.2f}%")
print(f"Yelp Polarity accuracy (10-Shot): {yelp_accuracy_few_shot * 100:.2f}%")

# Evaluate on AG News (multiclass classification) - Zero Shot and 10-Shot
ag_news_accuracy_zero_shot = evaluate_classification_model("ag_news", few_shot=False)
ag_news_accuracy_few_shot = evaluate_classification_model("ag_news", few_shot=True)
print(f"AG News accuracy (Zero Shot): {ag_news_accuracy_zero_shot * 100:.2f}%")
print(f"AG News accuracy (10-Shot): {ag_news_accuracy_few_shot * 100:.2f}%")

# Evaluate on TREC (multiclass classification) - Zero Shot and 10-Shot
trec_accuracy_zero_shot = evaluate_classification_model("trec", few_shot=False)
trec_accuracy_few_shot = evaluate_classification_model("trec", few_shot=True)
print(f"TREC accuracy (Zero Shot): {trec_accuracy_zero_shot * 100:.2f}%")
print(f"TREC accuracy (10-Shot): {trec_accuracy_few_shot * 100:.2f}%")
