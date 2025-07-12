import json
import os
import requests
import pandas as pd
import gradio as gr

API_SUBMIT_URL = "https://agents-course-unit4-scoring.hf.space/submit"

def submit_existing_answers(profile: gr.OAuthProfile | None, file_path: str):
    """
    Gradio callback: read uploaded JSON file at file_path,
    submit to the scoring API, and return status + a DataFrame.
    """
    # 1. Make sure user is logged in
    if profile is None:
        return "Please log in to Hugging Face first.", None

    # 2. Load the JSON from the uploaded file path
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
    except Exception as e:
        return f"Failed to load JSON: {e}", None

    # 3. Build the payload
    payload = {
        "username": profile.username,
        "agent_code": f"https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main",
        "answers": answers
    }

    # 4. Send request and handle errors
    try:
        response = requests.post(API_SUBMIT_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Submission failed: {e}", None

    # 5. Format successful status
    status = (
        f"Submission Successful!\n"
        f"User: {data.get('username')}\n"
        f"Score: {data.get('score','N/A')}% "
        f"({data.get('correct_count','?')}/{data.get('total_attempted','?')})"
    )
    # 6. Return status and a pandas DataFrame
    return status, pd.DataFrame(answers)

# Build Gradio interface
with gr.Blocks() as demo:
    # 1) Login button returns a gr.OAuthProfile
    login_btn = gr.LoginButton()

    # 2) File uploader (single JSON)
    upload = gr.File(label="Upload output.json", file_types=[".json"])

    # 3) Submit button
    submit_btn = gr.Button("Submit Uploaded Answers")

    # 4) Outputs
    status_out = gr.Textbox(label="Submission Status", lines=5, interactive=False)
    table_out  = gr.DataFrame(label="Answers Table")

    # 5) Wire up callback: pass login_btn (profile) and upload (file path)
    submit_btn.click(
        fn=submit_existing_answers,
        inputs=[upload],
        outputs=[status_out, table_out]
    )

    demo.launch()
