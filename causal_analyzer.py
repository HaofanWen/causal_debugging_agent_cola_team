# File_path: .\causal_analyzer.py
import json
import os
import threading
import zipfile
import magic   # python-magic: for MIME type detection
import pandas as pd
from pypdf import PdfReader
import docx
import pptx
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama_models import causal_analyzer

# Number of parallel threads; adjust to your hardware capacity
MAX_WORKERS = 2

def load_auxiliary_content(task_id: str, base_folder: str) -> str:
    """
    Look for a file named <task_id> with various possible extensions in base_folder.
    If found, attempt to extract text or metadata, returning a single string.
    Supported extensions (and strategies):
      - .pdf           → extract text from all pages via PdfReader
      - .docx          → extract paragraphs via python-docx
      - .pptx          → extract slide text via python-pptx
      - .xlsx / .xls   → read first sheet with pandas and convert to text
      - .py / .txt     → read file as plain UTF-8 text
      - .png / .jpg    → OCR via pytesseract (requires Tesseract installed)
      - .zip           → list contained filenames
      - .mp3           → return “Audio file present” (no transcription)
      - .pdb / .sls    → read as text if possible (otherwise note binary)
      - any other      → note “Unsupported file type: <mime>”
    If multiple matching extensions exist, priority is given in the order above.
    Returns an empty string if no auxiliary file is found.
    """
    # Define possible extensions and their handlers in priority order
    handlers = [
        (".pdf", _read_pdf_text),
        (".docx", _read_docx_text),
        (".pptx", _read_pptx_text),
        (".xlsx", _read_excel_text),
        (".xls", _read_excel_text),
        (".py", _read_plain_text),
        (".txt", _read_plain_text),
        (".png", _read_image_ocr),
        (".jpg", _read_image_ocr),
        (".jpeg", _read_image_ocr),
        (".zip", _read_zip_listing),
        (".mp3", _note_audio_file),
        (".pdb", _read_plain_or_note),
        (".sls", _read_plain_or_note),
    ]

    for ext, func in handlers:
        candidate = os.path.join(base_folder, f"{task_id}{ext}")
        if os.path.isfile(candidate):
            try:
                return func(candidate)
            except Exception as e:
                return f"[ERROR reading {task_id}{ext}: {e}]"

    # No auxiliary file found
    return ""


def _read_pdf_text(pdf_path: str) -> str:
    text_chunks = []
    with open(pdf_path, "rb") as f_pdf:
        reader = PdfReader.PdfReader(f_pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


def _read_docx_text(docx_path: str) -> str:
    doc = docx.Document(docx_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs).strip()


def _read_pptx_text(pptx_path: str) -> str:
    prs = pptx.Presentation(pptx_path)
    slide_texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_texts.append(shape.text)
    return "\n".join(slide_texts).strip()


def _read_excel_text(xlsx_path: str) -> str:
    df = pd.read_excel(xlsx_path, sheet_name=0)
    return df.to_string(index=False).strip()


def _read_plain_text(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _read_image_ocr(img_path: str) -> str:
    # Use Tesseract OCR to extract any text in the image
    img = Image.open(img_path).convert("RGB")
    ocr_text = pytesseract.image_to_string(img)
    return ocr_text.strip()


def _read_zip_listing(zip_path: str) -> str:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            namelist = z.namelist()
        return "ZIP file contents:\n" + "\n".join(namelist)
    except Exception as e:
        return f"[ERROR listing ZIP content: {e}]"


def _note_audio_file(mp3_path: str) -> str:
    # For audio, simply note that it exists. No transcription.
    return f"[Audio file present: {os.path.basename(mp3_path)}]"


def _read_plain_or_note(path: str) -> str:
    """
    Try to read as UTF-8 text. If fails, return a note about binary content.
    Used for .pdb, .sls, or other code-like/binary files.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except:
        return f"[Binary or unsupported content: {os.path.basename(path)}]"


def analyze_failure_stream(task_id: str, steps_text: str, aux_text: str, max_tokens: int = 512):
    """
    Perform a streaming causal-chain analysis for a single task,
    combining both the “Steps” text and any auxiliary content.

    Args:
        task_id: The unique ID of the task.
        steps_text: The “Steps” field from metadata.jsonl (string).
        aux_text: Text extracted from auxiliary file (empty if none).
        max_tokens: Maximum tokens to generate (default 512).

    Returns:
        A tuple (task_id, full_analysis_text).
    """
    client = causal_analyzer.client

    # Build the combined prompt
    prompt = (
        "Below is an Annotator “Steps” sequence and, if available, supplementary "
        "information extracted from an auxiliary file. Please produce a clear causal-chain analysis "
        "explaining how each step logically follows from the previous steps.\n\n"
        "=== Steps Begin ===\n"
        f"{steps_text.strip()}\n"
        "=== Steps End ===\n\n"
    )

    if aux_text:
        prompt += (
            "=== Auxiliary Document Content Begin ===\n"
            f"{aux_text.strip()}\n"
            "=== Auxiliary Document Content End ===\n\n"
        )

    prompt += (
        "Output format:\n"
        "1. [Top-Level Causal Conclusion]: One-sentence summary of the core inference.\n"
        "2. [Causal Chain]:\n"
        "   a) Step 1 → Explanation of why this step occurs.\n"
        "   b) Step 2 → Explanation of its logical connection to Step 1.\n"
        "   c) Step 3 → Explanation of its relationship to earlier steps.\n"
        "   ...\n"
        "Finally, include a brief summary highlighting the most important causal links "
        "and their implications."
    )

    # Send a streaming request
    response_stream = client.completion(
        model="ollama/nous-hermes2-mixtral:latest",
        provider="ollama",
        api_base="http://localhost:11434",
        api_key="ollama",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True
    )

    # Print header to mark the start of streaming output
    print(f"\n=== Task {task_id} Causal Analysis Start (Streaming) ===\n")
    full_text = ""

    for chunk in response_stream:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            full_text += content

    # Print footer after streaming ends
    print(f"\n\n=== Task {task_id} Causal Analysis End ===\n")
    return task_id, full_text


def process_validation_set_concurrent(jsonl_path: str):
    """
    Load validation metadata.jsonl, find any auxiliary file for each task_id,
    and run streaming causal analysis in parallel threads.

    Args:
        jsonl_path: Path to "GAIA_dataset/2023/validation/metadata.jsonl".
    """
    base_folder = os.path.dirname(os.path.abspath(jsonl_path))

    # Step 1: Read JSONL and collect (task_id, steps_text, aux_text)
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line.strip())
            task_id = record.get("task_id", "")
            steps_text = record.get("Annotator Metadata", {}).get("Steps", "").strip()
            if not steps_text:
                print(f"[WARNING] task_id={task_id} has no Steps; skipping.")
                continue

            # Load auxiliary content, if any
            aux_text = load_auxiliary_content(task_id, base_folder)
            tasks.append((task_id, steps_text, aux_text))

    if not tasks:
        print("No tasks found to process. Exiting.")
        return

    # A lock to avoid mixing streaming output from different threads
    print_lock = threading.Lock()

    def worker(task_tuple):
        task_id, steps_text, aux_text = task_tuple
        with print_lock:
            return analyze_failure_stream(task_id, steps_text, aux_text)

    # Step 2: Launch threads to process tasks concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(worker, t): t for t in tasks}

        for future in as_completed(future_to_task):
            task_id, analysis_text = future.result()
            # Optionally save to a file:
            with open("causal_outputs.jsonl", "a", encoding="utf-8") as fout:
                json.dump({"task_id": task_id, "causal_analysis": analysis_text}, fout)
                fout.write("\n")
            pass

    print("\nAll tasks processed.")


if __name__ == "__main__":
    # Assume this script is at project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "GAIA_dataset", "2023", "validation", "metadata.jsonl")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Cannot find validation file at: {metadata_path}")

    process_validation_set_concurrent(metadata_path)
