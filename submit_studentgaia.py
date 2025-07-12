import os
import json
import requests

# Base URL for the scoring API
API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"
SUBMIT_ENDPOINT = f"{API_BASE_URL}/submit"

# These environment variables are set automatically in your Hugging Face Space.
# Locally, you can export them yourself or hardcode for testing.
USERNAME = os.getenv("HF_USERNAME", "<your-username>")
SPACE_ID = os.getenv("SPACE_ID", "<your-username>/<your-space-name>")

def main():
    """
    Load the JSON array from answers/output.json, build the submission payload,
    post to the scoring API, and print the result.
    """
    # 1. Load answers
    with open("answers/output.json", encoding='utf-8') as f:
        answers = json.load(f)

    # 2. Build payload
    payload = {
        "username": USERNAME,
        "agent_code": f"https://huggingface.co/spaces/{SPACE_ID}/tree/main",
        "answers": answers
    }

    # 3. Submit and handle response
    try:
        response = requests.post(SUBMIT_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print("Submission failed:", e)
        return

    # 4. Display results
    print("Submission successful!")
    print(f" User:  {result.get('username')}")
    print(f" Score: {result.get('score','N/A')}% ({result.get('correct_count','?')}/{result.get('total_attempted','?')})")
    print(" Message:", result.get("message",""))

if __name__ == "__main__":
    main()
