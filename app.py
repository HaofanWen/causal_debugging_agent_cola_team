import json
import os
import requests
import pandas as pd
import gradio as gr

API_SUBMIT_URL = "https://agents-course-unit4-scoring.hf.space/submit"

def submit_existing_answers(profile: gr.OAuthProfile | None, file_obj):
    """
    Gradio callback: read uploaded JSON file of answers,
    submit to the scoring API, and return status + a DataFrame.
    """
    if not profile:
        return "Please log in to Hugging Face first.", None

    # Load the uploaded JSON
    answers = json.load(file_obj)

    payload = {
        "username": profile.username,
        "agent_code": f"https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main",
        "answers": answers
    }

    # Call the scoring API
    response = requests.post(API_SUBMIT_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Build status message
    status = (
        f"Submission Successful!\n"
        f"User: {data.get('username')}\n"
        f"Score: {data.get('score','N/A')}% "
        f"({data.get('correct_count','?')}/{data.get('total_attempted','?')})"
    )

    # Return status and a DataFrame showing the answers
    return status, pd.DataFrame(answers)

# Then in your Gradio Blocks:
with gr.Blocks() as demo:
    # ... your existing components ...
    upload = gr.File(label="Upload output.json", file_types=[".json"])
    submit_btn = gr.Button("Submit Uploaded Answers")
    status_out = gr.Textbox(label="Submission Status", lines=5, interactive=False)
    table_out  = gr.DataFrame(label="Answers Table")

    submit_btn.click(
        fn=submit_existing_answers,
        inputs=[gr.State(), upload],
        outputs=[status_out, table_out]
    )

    demo.launch()
