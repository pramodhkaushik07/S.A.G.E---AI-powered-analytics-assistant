from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict, Any
from openai import OpenAI
import os, io, re, json, base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from langchain_chroma import Chroma

# Initialize OpenAI client
client = OpenAI()
class State(TypedDict):
    messages: List[Dict[str, Any]]
    last_user_text: str
    query_type: Optional[str]
    chart_image: Optional[str]
    chart_summary: Optional[str]
    chart_context: Optional[str]
    chart_files: Optional[List[str]]
    sources: Optional[List[Dict[str, Any]]]


from vector_db import get_retriever
retriever = None

def get_lazy_retriever():
    print("[SYSTEM] Initializing retriever if not already loaded...")
    global retriever
    if retriever is None:
        print("[SYSTEM] Retriever loading now...")
        retriever = get_retriever()
        print("[SYSTEM] Retriever ready!")
    else:
        print("[SYSTEM] Retriever already loaded!")
    return retriever

#---------------------------- HELPER FUNCTION ---------------------------------


def append_user_message(state: State, text: str) -> State:
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
    state["messages"] = messages
    return state

def append_assistant_message(state: State, text: str) -> State:
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    state["messages"] = messages
    return state

# ----------------------------- CLASSIFIER NODE ------------------------------

def classifier_node(state: State) -> State:
    print("[SYSTEM] 🔄 Starting request processing…")
    """
    Purely semantic classifier — no keyword or pattern bias.
    Decides if a user query is conversational (chat),
    document-based (rag), or data-visual (visualization).
    """
    user_text = state["last_user_text"]
    print(f"[Classifier] Incoming user query: {user_text}")


    prompt = f"""
You are a reasoning router for Kestrel Education’s AI assistant.
The assistant can (1) chat naturally, (2) retrieve information from project
documents, or (3) visualize quantitative data.

Your task: infer the user's *intent* — not by keywords, but by purpose.

- **chat** → user wants a general discussion, greeting, or open advice that
  doesn’t depend on stored facts or datasets.

- **rag** → user seeks *knowledge retrieval*: factual, procedural, or policy
  information that exists inside documents (e.g., onboarding guide, project charter,
  communications plan). The goal is understanding or summarizing knowledge.

- **visualization** → user wants to *see* or *analyze* numerical information,
  such as comparisons, trends, distributions, or metrics.

Guidelines:
• Focus on the goal of the question, not its wording.
• If the answer requires looking up or recalling written material → rag.
• If it requires computing or plotting numbers → visualization.
• If it’s purely conversational or motivational → chat.

Query:
{user_text}

Return only one word: chat, rag, or visualization.
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0,
    )

    classification = resp.output_text.strip().lower()
    if classification not in {"chat", "rag", "visualization"}:
        classification = "chat"

    print(f"[Classifier] {user_text} → {classification}")
    state["query_type"] = classification
    return state

# ------------------------- CHAT NODE----------------------------------
def llm_node(state: State) -> State:
    user_text = state["last_user_text"]
    print(f"[Chat] Responding conversationally to: {user_text}")

    # Append user message (optional if already appended before graph)
    state = append_user_message(state, user_text)

    # Generate model response (no threading needed)
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=user_text,
        temperature=0.7,
    )
    # Append assistant reply
    state = append_assistant_message(state, resp.output_text)
    print("[Chat] Response ready")
    return state

# ----------------------- RAG NODE ------------------------------------
def rag_node(state: State) -> State:
    user_text = state["last_user_text"]
    print(f"[RAG] Searching documents for: {user_text}")

    hybrid_retriever = get_retriever(retriever_type="hybrid")
    print("[RAG] Retriever ready — invoking now...")


    # Step 1: Retrieve relevant documents
    retrieved_docs = hybrid_retriever.invoke(user_text) 
    print(f"[RAG] Retrieved {len(retrieved_docs)} relevant documents")

    state["sources"] = [
        { 
        "source": d.metadata.get("source"),
        "type": d.metadata.get("file_type"),
        "page": d.metadata.get("page") or d.metadata.get("page_number")
        }
        for d in retrieved_docs
    ]
                        
    
    if not retrieved_docs:
        print("[rag_node] No documents retrieved.")
        state = append_user_message(state, user_text)
        return append_assistant_message(state, "I couldn’t find that in the documents.")

    # Step 2: Combine context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Step 3: Build prompt
    prompt = f"""
You are a project assistant for Kestrel Education working on an admissions optimization project for GW University.

Use the following context to answer the user's question.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {user_text}

If the answer isn't in the context, say "I couldn’t find that in the documents."
Answer:
    """

    # Step 4: Query LLM
    try:
        print("[RAG] Sending context to LLM...")
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.3,
        )
        answer = resp.output_text.strip()
        print("[RAG] LLM successfully returned answer ")
    except Exception as e:
        print("[rag_node] LLM call failed:", e)
        answer = "Sorry, I couldn’t generate an answer right now."

    # Step 5: Update chat state
    state = append_user_message(state, user_text)
    state = append_assistant_message(state, answer)

    return state
# ---------------------------- CHART RAG NODE -------------------------------

def chart_rag_node(state: State) -> State:
    user_text = state["last_user_text"]
    print(f"[Chart-RAG] Resolving dataset for visualization request: {user_text}")

    from vector_db import get_retriever
    tabular_retriever = get_retriever(retriever_type="tabular")
    enhanced_query = f"{user_text} spreadsheet data table csv excel"
    print(f"[Chart-RAG] Enhanced query → {enhanced_query}")


    # Step 1: Retrieve
    retrieved_docs = tabular_retriever.invoke(enhanced_query)
    print(f"[Chart-RAG] Retrieved {len(retrieved_docs)} tabular candidates")
    state["sources"] = [
        { 
        "source": d.metadata.get("source"),
        "type": d.metadata.get("file_type"),
        "page": d.metadata.get("page") or d.metadata.get("page_number")
        }
        for d in retrieved_docs
    ]
    spreadsheet_docs = [
        doc for doc in retrieved_docs 
        if doc.metadata.get("source", "").lower().endswith((".csv", ".xlsx"))
    ]
    if not spreadsheet_docs:
        print(" [chart_rag_node] No tabular files found.")
        state["chart_context"] = ""
        state["chart_files"] = []
        return state

    # Step 2: Preview all tabular files (2–3 rows)
    print("[Chart-RAG] Previewing top candidate files...")
    base_dir = (
        "/content/drive/MyDrive/Synthetic_Data"
        if os.path.exists("/content/drive/MyDrive")
        else "/Users/pramodhkaushik/Desktop/Chartbot/Real_Data"
    )

    preview_blocks = []
    for doc in spreadsheet_docs:
        rel_path = doc.metadata.get("source", "")
        full_path = os.path.join(base_dir, rel_path)
        print(f"[Chart-RAG] Checking file: {full_path}")

        try:
            if rel_path.endswith(".csv"):
                df = pd.read_csv(full_path, nrows=3)
            else:
                df = pd.read_excel(full_path, nrows=3)
            df = normalize_columns(df)
            block = f"File: {rel_path}\nColumns: {df.columns.tolist()}\nPreview: {df.to_dict(orient='records')}\n"
            preview_blocks.append(block)
            print(f"[Chart-RAG] Previewed successfully: {rel_path}")
        except Exception as e:
            print(f" ⚠️ Error loading {rel_path}: {e}")

    if not preview_blocks:
        print("[Chart-RAG] No previews generated — aborting.")
        state["chart_context"] = ""
        state["chart_files"] = []
        return append_assistant_message(state, "I couldn't load any of the tabular files to preview.")

    combined_preview = "\n---\n".join(preview_blocks)
    print("[Chart-RAG] Asking LLM to pick best dataset...")

    # Step 3: Ask LLM to pick the best dataset
    prompt = f"""
You are a data analyst.

The user asked: "{user_text}"

Here are the available tabular files with their column names and a few rows:

{combined_preview}

Based on the user query, which ONE file is the most appropriate for visualization?

Return ONLY the file path, like:
Real_Data/admissions/2023_budget_summary.xlsx
"""

    resp = client.responses.create(model="gpt-4o", input=prompt, temperature=0)
    best_path = resp.output_text.strip()
    print(f"[Chart-RAG] LLM selected dataset → {best_path}")

    # Check if path is actually in list of retrieved files
    matched = [
        doc.metadata.get("source", "") for doc in spreadsheet_docs
        if doc.metadata.get("source", "").strip() == best_path
    ]
    if not matched:
        print("[Chart-RAG] LLM returned unmatched path. Stopping.")
        return append_assistant_message(state, f"I couldn't match the dataset to any available files.\nLLM suggested: `{best_path}`")

    # Set selected file and context
    state["chart_files"] = [best_path]
    state["chart_context"] = f"Best file selected: {best_path}"
    print(f" ✅ Best file selected: {best_path}")
    return state
# ----------------------------- VISUALIZATION NODE ------------------------------
import os, io, re, base64, traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from openai import OpenAI
import traceback


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to be LLM- and Python-friendly."""
    def clean(name):
        name = str(name).strip()
        name = re.sub(r"[^\w%]+", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")
    df.columns = [clean(c) for c in df.columns]
    return df


def filter_tabular_files(file_list):
    """Return only CSV and Excel files from a given list."""
    return [f for f in file_list if Path(f).suffix.lower() in [".csv", ".xlsx"]]
    
def chart_code_node(state: dict) -> dict:
    print("[Chart] Entered visualization node ")

    user_query = state["last_user_text"]
    context = state.get("chart_context", "")
    chart_files = state.get("chart_files", [])
    print(f"[Chart] User request → {user_query}")
    print(f"[Chart] Context → {context}")
    print(f"[Chart] Candidate files → {chart_files}")

    if not chart_files:
        print("[Chart]  No files available for plotting.")
        return append_assistant_message(state, "I couldn't find any dataset related to your question.")

    base_dir = (
        "/content/drive/MyDrive/Synthetic_Data"
        if os.path.exists("/content/drive/MyDrive")
        else "/Users/pramodhkaushik/Desktop/Chartbot/Real_Data"
    )
    selected_path = chart_files[0]
    if not os.path.isabs(selected_path):
        selected_path = os.path.join(base_dir, selected_path)

    print(f"[Chart]  Using dataset → {selected_path}")

    try:
        if selected_path.endswith(".csv"):
            df = pd.read_csv(selected_path)
        elif selected_path.endswith(".xlsx"):
            df = pd.read_excel(selected_path)
        else:
            return append_assistant_message(state, "Unsupported file type for visualization.")
        df = normalize_columns(df)
        print(f"[Chart] ✅ Loaded dataset. Columns → {df.columns.tolist()}")
    except Exception as e:
        print(f"[Chart] ❌ Error reading dataset → {e}")
        return append_assistant_message(state, f"Error reading dataset: {e}")

    # Debug snapshot
    print("\n" + "=" * 60)
    print("[Chart] DATASET SNAPSHOT")
    print(f"→ File: {selected_path}")
    print(f"→ Shape: {df.shape}")
    print(f"→ Columns: {df.columns.tolist()}")
    print("→ First 3 rows:")
    print(df.head(3).to_dict(orient="records"))
    print("=" * 60 + "\n")

    def run_code_and_save_image(code: str) -> None:
        """Exec chart code and populate state['chart_image']."""
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        exec(code, {"df": df, "plt": plt, "sns": sns, "pd": pd})
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        state["chart_image"] = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

    print("[Chart] Sending prompt to generate code + summary...")
    chart_prompt = f"""
You are a Python data visualization expert.

USER REQUEST:
{user_query}

AVAILABLE DATA:
- Columns: {df.columns.tolist()}
- Numeric columns: {df.select_dtypes(include="number").columns.tolist()}
- Categorical columns: {df.select_dtypes(exclude="number").columns.tolist()}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns

SAMPLE DATA (first 3 rows):
{df.head(3).to_dict('records')}

TASK:
1. Decide the most appropriate chart to answer the user's request.
2. Write Python code that uses the existing dataframe 'df' to build that chart.
3. Write a short, natural-language explanation that:
   - Clearly states what is being shown (metrics, dimensions, filters).
   - Highlights the main trend, comparison, or pattern.
   - Adds ONE light piece of analysis or implication (e.g., "this suggests...", "this indicates...").
   - Stays concise: about 3–5 sentences, no bullet points.

REQUIREMENTS FOR CODE:
- Use the dataframe 'df' (already loaded, do NOT reload it).
- Use ONLY columns that exist.
- Use seaborn + matplotlib (imported as sns, plt).
- Create a figure: plt.figure(figsize=(10, 6)).
- Add a clear title, axis labels, and legend if needed.
- DO NOT call plt.show().
- Do not print anything inside the code.

OUTPUT FORMAT:
Return ONLY valid JSON with exactly two keys: "code" and "summary".

- "code": a string containing ONLY the executable Python code (no backticks, no markdown).
- "summary": a short 3–5 sentence explanation in natural language, as if you are an analyst describing the chart under a dashboard. It must accurately describe the chart that the code will produce.

Example format (structure only, not content):
{{
  "code": "import matplotlib.pyplot as plt\\nimport seaborn as sns\\n... python code here ...",
  "summary": "This chart shows ... It highlights ... This suggests that ..."
}}
"""

    try:
        resp = client.responses.create(model="gpt-5", input=chart_prompt)
        raw = resp.output_text
        print("[Chart] Raw model output (truncated) →")
        print(raw[:400], "...")

        # Parse JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # If the model wrapped JSON in ``` blocks, strip them first
            json_blocks = re.findall(r"```(?:json)?(.*?)```", raw, re.DOTALL)
            if json_blocks:
                parsed = json.loads(json_blocks[0].strip())
            else:
                raise

        code = parsed.get("code", "").strip()
        summary = parsed.get("summary", "").strip()

        if not code:
            raise ValueError("No 'code' returned from model.")

        print("[Chart] ✅ Executing generated code...")
        run_code_and_save_image(code)

        if not summary:
            summary = f"Chart generated for your request: {user_query}"

        state["chart_summary"] = summary
        print(f"[Chart] ✅ SUCCESS. Summary → {summary}")
        return append_assistant_message(state, summary)

    except Exception as e:
        print(f"[Chart] ❌ EXECUTION/JSON ERROR → {e}")
        print(traceback.format_exc())

        # Fallback: simpler second attempt, still ask for code + summary
        retry_prompt = f"""
The previous attempt to generate code failed with error: {e}

USER REQUEST:
{user_query}

AVAILABLE COLUMNS:
{df.columns.tolist()}

SAMPLE DATA (first 2 rows):
{df.head(2).to_dict('records')}

Please return simpler code and a matching explanation.

REQUIREMENTS:
- Same as before (use df, seaborn+matplotlib, no plt.show()).
- Return ONLY JSON with "code" and "summary" keys.
"""

        try:
            retry_resp = client.responses.create(model="gpt-5", input=retry_prompt)
            raw2 = retry_resp.output_text
            print("[Chart] Retrying with simplified output (truncated) →")
            print(raw2[:400], "...")

            try:
                parsed2 = json.loads(raw2)
            except json.JSONDecodeError:
                json_blocks2 = re.findall(r"```(?:json)?(.*?)```", raw2, re.DOTALL)
                if json_blocks2:
                    parsed2 = json.loads(json_blocks2[0].strip())
                else:
                    raise

            retry_code = parsed2.get("code", "").strip()
            retry_summary = parsed2.get("summary", "").strip()

            if not retry_code:
                raise ValueError("No 'code' in retry response.")

            run_code_and_save_image(retry_code)

            if not retry_summary:
                retry_summary = f"Chart generated for your request: {user_query}"

            state["chart_summary"] = retry_summary
            print(f"[Chart] ✅ RECOVERED SUCCESS. Summary → {retry_summary}")
            return append_assistant_message(state, retry_summary)

        except Exception as e2:
            print(f"[Chart] ❌ RETRY FAILED → {e2}")
            print(traceback.format_exc())
            return append_assistant_message(state, "Chart generation failed — please clarify your request.")



       
# ---------------------------- GRAPH -------------------------------
builder = StateGraph(State)
builder.add_node("classifier", classifier_node)
builder.add_node("chat", llm_node)
builder.add_node("rag", rag_node)
builder.add_node("chart_rag", chart_rag_node)
builder.add_node("visualization", chart_code_node)

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier",
    lambda s: s.get("query_type"),
    {"chat": "chat", "rag": "rag", "visualization": "chart_rag"}
)
builder.add_edge("chart_rag", "visualization")
builder.add_edge("visualization", END)
builder.add_edge("chat", END)
builder.add_edge("rag", END)

app = builder.compile()

# ------------------------ TEST -----------------------------------
def run_debug_test(queries: List[str]):
    for q in queries:
        print(f"\nQUERY: {q}")
        print("=" * 70)
        state = {"last_user_text": q, "messages": []}
        for step in app.stream(state):
            print(step)
        result = app.invoke(state)
        msgs = [m["content"][0]["text"] for m in result.get("messages", []) if m["role"]=="assistant"]
        if msgs: print("\nAssistant:", msgs[-1])
        if result.get("chart_image"): print(" Chart image generated.")
        print("=" * 70)

if __name__ == "__main__":
    test_queries = [
        "Hello, what can you do?",
        "What should a consultant do in the first 48 hours?",
        "Show me the plot of allocation % of each role in week 1.",
        "Summarize the budget and show a pie chart."
    ]
    run_debug_test(test_queries)
