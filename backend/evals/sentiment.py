from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

dataset_name = "Sentiment Analysis Dataset"
dataset = client.create_dataset(dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {"text": "I love this product!"},
            "outputs": {"sentiment": "positive"}
        },
        {
            "inputs": {"text": "This is the worst thing I ever bought."},
            "outputs": {"sentiment": "negative"}
        },
        {
            "inputs": {"text": "It's okay, not great but not terrible either."},
            "outputs": {"sentiment": "neutral"}
        },
        {
            "inputs": {"text": "I feel fantastic!"},
            "outputs": {"sentiment": "positive"}
        },
        {
            "inputs": {"text": "I'm so angry at you right now."},
            "outputs": {"sentiment": "negative"}
        },
        {
            "inputs": {"text": "I'm not sure how I feel about this."},
            "outputs": {"sentiment": "neutral"}
        }
    ]
)

def check_correctness(inputs: dict, outputs: dict, ref_outputs: dict) -> bool:
    for output, ref_output in zip(outputs.values(), ref_outputs.values()):
        if output.strip().lower() != ref_output.strip().lower():
            return False
    return True

def check_conciseness(outputs: dict, ref_outputs: dict) -> bool:
    allowed_responses = {'positive', 'negative', 'neutral'}

    for output in outputs.values():
        if output.strip().lower() not in allowed_responses:
            return False
    return True

def my_app(text: str) -> str:
    import requests
    url = f"http://0.0.0.0:8001/v1/sentiment"

    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    response = requests.post(url, json=payload, headers=headers)
    return response.json()['content']

def langsmith_target(inputs: dict) -> dict:
    return {'response': my_app(inputs['text'])}

experiment_results = client.evaluate(
    langsmith_target,
    data=dataset_name,
    evaluators=[check_correctness, check_conciseness],
    experiment_prefix="Llama-3.2-1B-Instruct-Sentiment",
)