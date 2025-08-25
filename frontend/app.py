import gradio as gr
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", None)
API_BASE_URL = os.getenv("BACKEND_API_URL", None)

if not API_BASE_URL or not API_KEY:
    raise ValueError("Please set the BACKEND_API_URL and OPENAI_API_KEY environment variables.")


def analyze_sentiment(text):
    """
    Send a POST request to the sentiment analysis API.
    """
    if not text.strip():
        return "Empty text provided"

    url = f"{API_BASE_URL}/sentiment"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 422:
            return "Error: Backend validation failed. Please check your input."

        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text[:200]}"

        try:
            result = response.json()
            return f'Sentiment: {result.get("content", "unknown")}'
        except json.JSONDecodeError:
            return f'Error: Failed to parse response from backend. Response text: {response.text[:200]}'
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to the backend service."
    except requests.Timeout:
        return "Error: The request timed out."
    except requests.RequestException as e:
        return f"Error: An unexpected error occurred: {e}"

def answer_medical_question(question, choice_a, choice_b, choice_c, choice_d):
    """
    Send a POST request to the medical question answering API.
    """
    if not question.strip():
        return "Empty question provided"

    url = f"{API_BASE_URL}/medical-qa"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "choices": [choice_a, choice_b, choice_c, choice_d]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 422:
            return "Error: Backend validation failed. Please check your input."

        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text[:200]}"

        try:
            result = response.json()
            return result.get("content", "unknown")
        except json.JSONDecodeError:
            return f'Error: Failed to parse response from backend. Response text: {response.text[:200]}'
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to the backend service."
    except requests.Timeout:
        return "Error: The request timed out."
    except requests.RequestException as e:
        return f"Error: An unexpected error occurred: {e}"


###### UI ######

# Sentiment Analysis tab
with gr.Blocks() as sentiment_tab:
    gr.Markdown("# Sentiment Analysis")
    gr.Markdown("Enter text to analyze its sentiment (positive, negative, neutral).")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", lines=5, placeholder="Enter text here...")
            analyze_btn = gr.Button("Analyze Sentiment")
        with gr.Column():
            sentiment_output = gr.Textbox(label="Sentiment Result", lines=2)
        
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=[text_input],
        outputs=[sentiment_output]
    )

    gr.Examples(
        [
            ["I love this product!"],
            ["This is the worst experience I've ever had."],
            ["It's okay, not great but not terrible."],
            ["I'm not sure how I feel about this."],
            ["Let's go!!!"],
            ["Alright!"],
        ],
        inputs=[text_input],
        outputs=[sentiment_output],
        fn=analyze_sentiment,
    )

# Medical Question Answering tab
with gr.Blocks() as medical_qa_tab:
    gr.Markdown("# Medical Question Answering")
    gr.Markdown("Enter a medical question and four answer choices. The model will select the most appropriate answer.")

    question_input = gr.Textbox(label="Medical Question", lines=2, placeholder="Enter the medical question here...")

    with gr.Row():
        choice_a = gr.Textbox(label='Choice A', lines=1)
        choice_b = gr.Textbox(label='Choice B', lines=1)
    with gr.Row():
        choice_c = gr.Textbox(label='Choice C', lines=1)
        choice_d = gr.Textbox(label='Choice D', lines=1)

    answer_btn = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer", lines=2)

    answer_btn.click(
        fn=answer_medical_question,
        inputs=[question_input, choice_a, choice_b, choice_c, choice_d],
        outputs=[answer_output]
    )

    gr.Examples(
        [
            ["Axonal transport is:", "Antegrade", "Retrograde", "Antegrade and retrograde", "None"],
            ["Low insulin to glucagon ratio is seen in all of these except:", "Glycogen synthesis", "Glycogen breakdown", "Gluconeogenesis", "Ketogenesis"],
        ],
        inputs=[question_input, choice_a, choice_b, choice_c, choice_d],
        outputs=[answer_output],
        fn=answer_medical_question
    )


demo = gr.TabbedInterface(
    [sentiment_tab, medical_qa_tab],
    ["Sentiment Analysis", "Medical QA"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)