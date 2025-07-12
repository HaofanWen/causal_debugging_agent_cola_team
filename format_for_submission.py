import os
import json

# Directory where individual answer text files live
ANSWERS_DIR = os.path.join(os.path.dirname(__file__), 'answers')
OUTPUT_PATH = os.path.join(ANSWERS_DIR, 'output.json')

def main():
    """
    Read all answer_*.txt files, build a list of {task_id, submitted_answer},
    and write them as one JSON array to output.json.
    """
    submission_list = []

    # Iterate over sorted files like answer_1001.txt, answer_1002.txt, ...
    for filename in sorted(os.listdir(ANSWERS_DIR)):
        if filename.startswith("answer_") and filename.endswith(".txt"):
            task_id = filename[len("answer_"):-4]  # strip prefix/suffix
            file_path = os.path.join(ANSWERS_DIR, filename)

            # Read the answer text
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                answer_text = f.read().strip()

            submission_list.append({
                "task_id": task_id,
                "submitted_answer": answer_text
            })

    # Write the list as a JSON array
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_file:
        json.dump(submission_list, out_file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(submission_list)} entries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
