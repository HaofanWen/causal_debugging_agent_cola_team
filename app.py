import json
import os
import requests
import pandas as pd
import gradio as gr

# Scoring API endpoint
API_SUBMIT_URL = "https://agents-course-unit4-scoring.hf.space/submit"

def submit_existing_answers(file_path: str):
    """
    Gradio callback: read the uploaded JSON file at file_path,
    submit its contents to the scoring API, and return a status message
    and a table of the answers.
    """
    # Load answers from the JSON file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
    except Exception as e:
        return f"Failed to load JSON: {e}", None

    # Determine the username (from env or placeholder)
    username = os.getenv("HF_USERNAME", "<your-username>")

    # Build payload for submission
    payload = {
        "username":   username,
        "agent_code": f"https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main",
        "answers":    answers
    }

    # Send request to scoring API
    try:
        response = requests.post(API_SUBMIT_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Submission failed: {e}", None

    # Format a user-friendly status message
    status = (
        f"Submission Successful!\n"
        f"User: {data.get('username')}\n"
        f"Score: {data.get('score','N/A')}% "
        f"({data.get('correct_count','?')}/{data.get('total_attempted','?')})\n"
        f"Message: {data.get('message','')}"
    )

    # Return the status and a DataFrame of the submitted answers
    return status, pd.DataFrame(answers)

# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Submit Pre-Generated Answers")
    gr.Markdown(
        """
        Upload your `output.json` (the JSON array of `{task_id, submitted_answer}` objects),
        then click **Submit Uploaded Answers** to get your score.
        """
    )

    # File uploader for a single JSON file
    upload     = gr.File(label="Upload output.json", file_types=[".json"])
    # Button to trigger submission
    submit_btn = gr.Button("Submit Uploaded Answers")
    # Textbox to display status
    status_out = gr.Textbox(label="Submission Status", lines=5, interactive=False)
    # Table to display the answers
    table_out  = gr.DataFrame(label="Answers Table")

    # Wire up the callback: only the uploaded file is passed as input
    submit_btn.click(
        fn=submit_existing_answers,
        inputs=[upload],
        outputs=[status_out, table_out]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
