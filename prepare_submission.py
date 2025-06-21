import os
import json
from evaluate_predictions import extract_core_answer

USERNAME = "HaofanWen"
SPACE_URL = "https://huggingface.co/spaces/HaofanWen/causal_debugging_agent/tree/main"
ANSWER_DIR = "./output"

answers = []

for filename in os.listdir(ANSWER_DIR):
    if filename.startswith("answer_") and filename.endswith(".txt"):
        task_id = filename[len("answer_"):-len(".txt")]
        with open(os.path.join(ANSWER_DIR, filename), "r", encoding="utf-8") as f:
            raw = f.read().strip()
            cleaned = extract_core_answer(raw)
            answers.append({
                "task_id": task_id,
                "submitted_answer": cleaned
            })

submission = {
    "username": USERNAME,
    "agent_code": SPACE_URL,
    "answers": answers
}

with open("gaia_submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2, ensure_ascii=False)

print(f"✅ Ready to submit: {len(answers)} answers written to gaia_submission.json")
