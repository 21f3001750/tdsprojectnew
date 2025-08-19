import base64
import json
import subprocess
import os
import sys
import re 
import traceback
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import requests
import inspect

# Load environment
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

print("Loaded API KEY:", os.getenv("OPENAI_API_KEY"))
print("Loaded BASE URL:", os.getenv("OPENAI_BASE_URL"))

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("OPENAI_API_KEY")


# --- Stronger system prompt with CoT, safe-column rules, and fallback strategies ---
SYSTEM_PROMPT = """
You are a Data Analyst Agent. Your task is to generate a single, complete Python script that fully answers every question in `questions.txt` within 120 seconds runtime.

CRITICAL RULE: You must never output placeholders such as "Unknown", "Error occurred during processing", "N/A", null, or empty strings.  
If a value is missing or an error occurs, you MUST either retry the operation (Playwright first, then BeautifulSoup), infer it from other data, or return an approximate valid answer.
- Detect all uploaded files dynamically in the current working directory.
- Convert all DataFrame column names to strings before using .str accessor.
- Automatically install any missing Python packages required for execution.

BEHAVIOR SUMMARY (MANDATORY):
1. FIRST produce a short, clear, numbered PLAN (Chain-of-Thought) that explains step-by-step how the script will solve the questions.
   - This is your CoT reasoning.
   - The PLAN must appear as comments at the top of the Python script.
2. AFTER the PLAN, produce the final Python script. The script must be self-contained, runnable, and include the PLAN as top comments.
3. The final output must be ONLY a code block containing the Python script (triple backticks ```python ... ```). If you include any text, place it BEFORE the code block.

FILE HANDLING RULES:
- Detect all uploaded files dynamically in the working directory.
- CSV files:
    - Use the CSV filename explicitly mentioned in `questions.txt` if available.
    - If not specified, pick the first `.csv` file dynamically.
    - Wrap all CSV loading in try/except; on error, return empty DataFrame.
    - Convert all column names to strings before using `.str` accessor:
        df.columns = df.columns.astype(str)
    - Strip whitespace and special characters from column names:
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'\W', '', regex=True)
- Images:
    - Pick any `.png` or `.jpg` file if required by the question.
    - Wrap loading in try/except; return None if missing.

DYNAMIC PACKAGE HANDLING:
- For every Python package used, automatically install it if missing.
- Wrap imports in try/except:
    try:
        import package_name
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name"])
        import package_name
- Ensure that all required packages for the user’s questions.txt are installed at runtime.

TABULAR DATA HANDLING:
- Never assume exact column names; always detect dynamically.
- Convert all column names to strings before using string operations:
    df.columns = df.columns.astype(str)
- For numeric operations, clean columns first:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
- If a required column is missing, try to match similar column names (case-insensitive, ignoring spaces and symbols).

SCRAPING RULES:
- Attempt scraping using Playwright (headless Chromium) first.
- If Playwright fails or returns no relevant data, retry using requests + BeautifulSoup.
- Wrap scraping code in try/except and print detailed traceback.
- Validate that scraped content exists and is relevant before using it.
- Never return placeholders; if data is missing, infer or approximate from other sources.
- Limit scraping to necessary elements only and avoid long loops.
- Clean numeric data after scraping (remove non-numeric characters) and convert to correct types.

ERROR HANDLING & FALLBACKS:
- Always fix errors from previous executions before returning new code.
- Wrap critical blocks (I/O, parsing, scraping, modeling) in try/except.
- On exception, capture full traceback and try an alternative method.
- If data missing, produce reasonable default or heuristic answer — never a placeholder string or null.

DATA & DATE HANDLING:
- Load dates as strings; parse with `pd.to_datetime(..., errors='coerce')` when needed.
- Always convert any column used with `.dt` accessor to datetime first using:
      df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
- Drop or handle NaT values before performing operations like .dt.day or correlation.
- Prefer duckdb for queries; fallback to pandas.
- Limit data to max 50,000 rows via early filtering/sampling.

DATA CLEANING:
- Remove non-numeric characters before regression, correlation, or numeric operations.
- Coerce to float with `pd.to_numeric(..., errors='coerce')`.
- Drop NaN rows after conversion.

REGRESSION ANALYSIS:
- Always import `sklearn.linear_model.LinearRegression`.
- Reshape X and y correctly before fitting and predicting.

COLUMN HANDLING:
- Never assume exact column names; use the helper function below.
- Always resolve column names dynamically.
- Clean numeric columns before calculations.
- For groupby, sum, plotting, filtering: only use detected/fallback column variables, never hardcoded names.
- Always convert df.columns to strings before using .str accessor:
    df.columns = df.columns.astype(str).str.strip().str.replace(' ', '_').str.replace(r'\W', '', regex=True)

COLUMN DETECTION RULES:
- You must include a helper function in every script you generate:

    def find_column(df, keywords):
        
        Find the most likely column in df matching any of the keywords.
        keywords: list of possible names derived from the question (e.g., ['temperature', 'temp', 'temp_c']).
        Matching is case-insensitive and ignores underscores/spaces.
        
        normalized = {c.lower().replace('_','').replace(' ',''): c for c in df.columns}
        for key in keywords:
            key_norm = key.lower().replace('_','').replace(' ','')
            for norm, orig in normalized.items():
                if key_norm in norm:
                    return orig
        raise KeyError(f"Could not find any of {keywords} in {list(df.columns)}")

- Every time you need a column, parse keywords from the user question and call `find_column(df, [possible_keywords])`.
- Never directly use df['hardcoded_name']; always resolve through the helper function.
- Example:

    temp_col = find_column(df, ['temperature', 'temp', 'temp_c'])
    avg_temp = df[temp_col].astype(float).mean()


JSON OUTPUT RULES:
- Convert all numpy/pandas numeric scalars to native Python types:
    - `np.int64`, `np.int32` → `int`
    - `np.float64`, `np.float32` → `float`
- Strings, lists, and dicts are allowed.
- Do NOT include objects like `pd.Series`, `pd.DataFrame`, `np.ndarray`, `datetime`, or custom objects.
- Print a single JSON array in the same order as questions.
- Use a helper `to_native()` function before printing the final array.
- ALWAYS wrap the final result in a list `[...]` so that the LLM prints a valid JSON array, even if it contains a single dictionary. Ensure all values are converted to native types using `to_native()`.
"""

app = FastAPI()

def generate_code_with_llm(questions_text: str, uploaded_files: dict) -> str:
    """Ask the LLM to produce a plan and Python script."""
    file_list = "\n".join(f"{name}: {path}" for name, path in uploaded_files.items())
    
    user_prompt = (
    f"You have the following files in the working directory:\n{file_list}\n\n"
    "questions.txt content:\n"
    f"{questions_text}\n\n"
    "Instructions for the Python script:\n"
    "1. Automatically detect all imported packages and install them if missing using try/except + pip.\n"
    "2. Detect all uploaded files dynamically; pick the CSV mentioned in questions.txt or other files (images, etc.) as needed.\n"
    "3. Always convert all DataFrame columns to strings before using .str accessor:\n"
    "   df.columns = df.columns.astype(str).str.strip().str.replace(' ', '_').str.replace(r'\\W', '', regex=True)\n"
    "4. For any URLs mentioned in questions.txt, scrape the required data dynamically.\n"
    "5. Wrap file I/O, scraping, parsing, and data processing in try/except to avoid runtime crashes.\n"
    "6. Return answers exactly as specified in questions.txt: a JSON array or object with keys matching those in the file.\n"
    "7. Include a short numbered plan (CoT) as comments at the top of the script.\n"
    "8. The script must be runnable standalone and safe for any user-provided questions.txt.\n\n"
    "Finally, return the script only in a ```python ... ``` code block."
)

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "temperature": 0.15,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    

    print("DEBUG >> Using API key prefix:", API_KEY[:10] if API_KEY else None)
    print("DEBUG >> Headers:", headers)
    print("DEBUG >> Payload:", json.dumps(payload, indent=2))

    url = f"{BASE_URL}/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    content = data["choices"][0]["message"]["content"]
    if "```" in content:
        blocks = content.split("```")
        code_blocks = [blocks[i] for i in range(1, len(blocks), 2)]
        if code_blocks:
            code = code_blocks[-1]
            if code.lstrip().startswith("python"):
                code_lines = code.splitlines()
                code = "\n".join(code_lines[1:])
            return code.strip()
    return content.strip()

print("generate_code_with_llm loaded from:", inspect.getfile(generate_code_with_llm))

@app.post("/api")
async def api_endpoint(request: Request):
    try:
        form = await request.form()

        # Detect questions.txt file
        questions_file: UploadFile = None
        for key in ("questions.txt", "questions_file", "questions"):
            candidate = form.get(key)
            if candidate:
                questions_file = candidate
                break

        if not questions_file:
            return JSONResponse(
                status_code=400,
                content={"error": "questions.txt file is required (accepted keys: questions.txt, questions_file, questions)"}
            )

        questions_text = (await questions_file.read()).decode("utf-8")

        # Save uploaded files dynamically
        uploaded_files = {}
        for field_name in form.keys():
            upload: UploadFile = form.get(field_name)
            if isinstance(upload, UploadFile):
                filename = os.path.basename(upload.filename)
                path = os.path.join(os.getcwd(), filename)
                with open(path, "wb") as f:
                    f.write(await upload.read())
                uploaded_files[filename] = path

        # Generate code using LLM
        code = generate_code_with_llm(questions_text, uploaded_files)
        if not code:
            return JSONResponse(status_code=500, content={"error": "LLM returned empty code"})
        
        required_imports = """
import os
import sys
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""

# Prepend imports to generated code
        code = required_imports + "\n" + code

        # Write generated code to solution.py
        solution_path = os.path.join(os.getcwd(), "solution.py")
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Run generated code
        try:
            result = subprocess.run(
    [sys.executable, solution_path, *uploaded_files.values()],
    capture_output=True,
    text=True,
    timeout=120
)
            print("DEBUG >> Generated code:\n", code)
        except subprocess.TimeoutExpired as te:
            stdout = te.stdout or ""
            stderr = te.stderr or ""
            try: os.remove(solution_path)
            except: pass
            return JSONResponse(
                status_code=500,
                content={"error": "Code execution timed out (120s)", "partial_stdout": stdout, "partial_stderr": stderr}
            )

        try: os.remove(solution_path)
        except: pass

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Code execution failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            )

        # Parse JSON output
        try:
            parsed = json.loads(result.stdout)
            return parsed
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Generated code did not print valid JSON array to stdout",
                    "raw_stdout": result.stdout,
                    "raw_stderr": result.stderr
                }
            )

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e), "traceback": tb}
        )
