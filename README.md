---
title: Causal Debugging Agent
emoji: 🐢
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.34.1
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



# Causal Debugging Agent

This project provides a causal debugging agent built on LangChain, LangGraph, and Gradio. It includes:

1. **Interactive Debugging Agent**  
   - Perform causal analysis or code repair on input problems via a Gradio interface.

2. **Batch Generation Script**  
   - Run the agent in batch mode to process a dataset of bugs.

3. **Evaluation Scripts**  
   - Compute Exact Match, BLEU, and AST-based metrics on the generated fixes.

---

## Requirements

- Python 3.9 or higher  
- It is recommended to use a virtual environment (venv or conda)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/HaofanWen/causal_debugging_agent_cola_team.git
cd causal_debugging_agent_cola_team

# Install Python dependencies
pip install -r requirements.txt
````

---

## Environment Variables

Create a `.env` file in the project root with the following keys:

```dotenv
# (Optional) Hugging Face username for Gradio sharing
HF_USERNAME=your_hf_username

# (Optional) Hugging Face Space ID
SPACE_ID=your_space_id

# Supabase database credentials
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_KEY=your_service_key
```

---

## Usage

### 1. Launch the Gradio Interface

```bash
python app.py
```

* Go to the displayed URL in your browser.
* Upload an `output.json` file and click **Submit Uploaded Answers** to view scores and answer tables.

### 2. Run Batch Generation

```bash
python main.py
```

* By default, reads from `bug_data/debug_dataset.jsonl` and writes results to `output.json`.
* To switch to the student dataset, uncomment and adjust the `INPUT_PATH` in `main.py`.

### 3. Evaluate Outputs

```bash
python debug_evaluate.py
```

* Computes Exact Match, BLEU, and AST scores and saves them in `evaluate_debug.json`.

---

## Project Structure

```
.
├── app.py                   # Gradio interface
├── main.py                  # Batch generation script
├── debug_evaluate.py        # Evaluation script
├── bug_data/                # Debug dataset (JSONL)
├── student_data/            # Student dataset (JSONL)
├── output.json              # Agent outputs
├── evaluate_debug.json      # Evaluation results
└── requirements.txt         # Python dependencies
```

---

## License

This project is licensed under the Apache License 2.0.
