import os
import json
import gradio as gr
from evaluate_predictions import extract_core_answer

# File path definitions
ANSWER_DIR = "./output"
ANALYSIS_FILE = "./causal_outputs.jsonl"

# Preload all causal analysis
causal_map = {}
if os.path.exists(ANALYSIS_FILE):
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            causal_map[record["task_id"]] = record["causal_analysis"]

def load_answer(task_id):
    path = os.path.join(ANSWER_DIR, f"answer_{task_id}.txt")
    if not os.path.exists(path):
        return "", "❌ Answer file not found", "❌ Causal analysis not found"
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    cleaned = extract_core_answer(raw)
    analysis = causal_map.get(task_id, "⚠️ No corresponding analysis record")
    return cleaned, raw, analysis

def export_submission():
    answers = []
    for fname in os.listdir(ANSWER_DIR):
        if fname.startswith("answer_") and fname.endswith(".txt"):
            task_id = fname[len("answer_"):-len(".txt")]
            with open(os.path.join(ANSWER_DIR, fname), "r", encoding="utf-8") as f:
                raw = f.read().strip()
            cleaned = extract_core_answer(raw)
            answers.append({"task_id": task_id, "submitted_answer": cleaned})

    submission = {
        "username": "HaofanWen",
        "agent_code": "https://huggingface.co/spaces/HaofanWen/causal_debugging_agent/tree/main",
        "answers": answers
    }

    with open("gaia_submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    return f"✅ Exported {len(answers)} answers to gaia_submission.json"

with gr.Blocks(title="GAIA Agent Debugger") as demo:
    gr.Markdown("## 🎯 GAIA Agent Debugging Interface\nEnter a task_id to view the answer and causal chain analysis")

    with gr.Row():
        task_input = gr.Text(label="Enter Task ID (e.g. fe8f4748-5d00-4a27-9070-090a0cfdeac4)")
        submit_btn = gr.Button("Search")

    with gr.Row():
        final_answer = gr.Text(label="🌟 Extracted Final Answer")
        raw_answer = gr.Text(label="📝 Raw Answer Text")

    analysis_output = gr.Text(label="📘 Causal Analysis", lines=10)

    submit_btn.click(fn=load_answer, inputs=[task_input],
                     outputs=[final_answer, raw_answer, analysis_output])

    gr.Markdown("---")
    with gr.Row():
        export_button = gr.Button("📥 Generate Submission JSON (gaia_submission.json)")
        export_status = gr.Textbox(label="Export Status")

    export_button.click(fn=export_submission, outputs=[export_status])

demo.launch()
