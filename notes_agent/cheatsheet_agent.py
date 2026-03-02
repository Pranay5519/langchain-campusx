"""
Cheat Sheet Agent using LangChain + Ollama
------------------------------------------
- User provides a topic or raw notes as input
- LLM (via Ollama) generates a beautiful HTML+CSS cheat sheet
- Output is saved as an HTML file
- Sequential workflow: Input -> Plan -> Generate HTML -> Save File
- No memory, no complex tools — keep it simple
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import os
import re
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
# ─────────────────────────────────────────
# 1. MODEL SETUP
# ─────────────────────────────────────────

# Change model name to any Ollama model you have pulled
# e.g. "llama3", "mistral", "qwen2", "gemma2"
# MODEL_NAME = "deepseek-coder:latest"

# llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# ─────────────────────────────────────────
# 2. PROMPTS
# ─────────────────────────────────────────

# STEP 1 PROMPT: Plan the sections from user input
plan_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a study notes planner. Read the user's input carefully.

User Input:
{user_input}

Your job:
1. Identify the main TOPIC name (short, 2-5 words)
2. Identify 3 to 6 key SECTIONS from the content
3. For each section write a short one-line summary

Respond in this EXACT format, nothing else:

TOPIC: <topic name>
SECTIONS:
- <Section Title> | <one line summary>
- <Section Title> | <one line summary>
- <Section Title> | <one line summary>
"""
)

# STEP 2 PROMPT: Generate the full HTML cheat sheet
html_prompt = PromptTemplate(
    input_variables=["user_input", "plan"],
    template="""
You are an expert HTML and CSS designer who creates beautiful cheat sheets.

Original Content:
{user_input}

Planned Structure:
{plan}

Your task: Generate a COMPLETE, self-contained HTML file that is a visual cheat sheet.

STRICT DESIGN RULES:
- Single HTML file with all CSS inside a <style> tag — no external files
- Use Google Fonts: import Playfair Display (headings) and DM Sans (body) via @import
- Background color: #faf6f0 (warm paper), Text: #1a1410 (dark ink)
- Each section card has a unique colored top border (use these colors in order: #c94f2a, #2a6fc9, #2ab87a, #b87a2a, #7a2ab8, #c92a6f)
- Large faded section numbers (01, 02...) behind each section title
- Cards grid layout (2 or 3 columns) for facts and bullet points
- Flow diagram (flexbox row with arrows →) for any process or steps
- Highlighted callout box for key definitions or examples
- Subtle fade-in animation on load using CSS @keyframes
- Compact, dense layout — this is a cheat sheet, not a long article
- Footer with topic name

CONTENT RULES:
- NEVER use raw bullet lists — convert all bullets into styled cards
- Bold all key terms using <strong>
- Every sequence or process must be a visual flow: Step1 → Step2 → Step3
- Examples get a callout block with an "EXAMPLE" label

OUTPUT RULES:
- Return ONLY the raw HTML code
- Start with <!DOCTYPE html> and end with </html>
- Do NOT add any explanation, markdown, or code fences
- Do NOT include ```html or ``` anywhere
"""
)


# ─────────────────────────────────────────
# 3. CHAINS (simple LLMChain — sequential)
# ─────────────────────────────────────────

plan_chain = LLMChain(llm=llm, prompt=plan_prompt, verbose=True)
html_chain = LLMChain(llm=llm, prompt=html_prompt, verbose=True)


# ─────────────────────────────────────────
# 4. HELPER: Clean LLM output
# ─────────────────────────────────────────

def clean_html(raw: str) -> str:
    """Remove any markdown fences the LLM might add."""
    raw = raw.strip()
    # Remove ```html ... ``` or ``` ... ```
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


# ─────────────────────────────────────────
# 5. MAIN AGENT FUNCTION
# ─────────────────────────────────────────

def run_cheatsheet_agent(user_input: str, output_file: str = "cheatsheet_output.html"):
    """
    Sequential agent:
      Node 1: Plan  → understand topic and sections
      Node 2: Generate → create full HTML cheat sheet
      Node 3: Save  → write HTML file to disk
    """

    print("\n" + "="*50)
    print("  CHEAT SHEET AGENT STARTING")
    print("="*50)

    # ── NODE 1: PLAN ──────────────────────────────
    print("\n[Node 1] Planning sections from your input...")
    plan_result = plan_chain.invoke({"user_input": user_input})
    plan_text = plan_result["text"].strip()
    print("\nPlan generated:")
    print(plan_text)

    # ── NODE 2: GENERATE HTML ─────────────────────
    print("\n[Node 2] Generating HTML cheat sheet...")
    html_result = html_chain.invoke({
        "user_input": user_input,
        "plan": plan_text
    })
    raw_html = html_result["text"]
    clean = clean_html(raw_html)

    # Basic sanity check
    if not clean.lower().startswith("<!doctype"):
        print("\n⚠️  Warning: Output may not be valid HTML. Check the file.")

    # ── NODE 3: SAVE FILE ─────────────────────────
    print(f"\n[Node 3] Saving cheat sheet to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(clean)

    print("\n✅ Done! Open the HTML file in your browser.")
    print(f"   File: {os.path.abspath(output_file)}")
    print("="*50 + "\n")

    return output_file


# ─────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║   CHEAT SHEET GENERATOR — Ollama Agent   ║")
    print("╚══════════════════════════════════════════╝")

    user_input = """
   langChain components 
--> Models prompts chains memory indexes agents
LangChain models
== LLMS chat models vision models hugging face open sourse models embedding models
prompts in Langchain 
==static dynamic , single message a list of messages prompt template chat prompt template message placeholder human messages system message AI message
structured output,  output parsers
== Structured output with structured output pyridic output parser Jason output parts are string output parser pigantic models nested pydantic models , list of models that supports which structure output
Chains in lc
== Sequential chain parallel chain conditional chain
Runnables
== task specific Runnables runnable primitives Renewable parallel Runnable Sequence Runnable pass through Runable branch runnable Lambda
document loaders
== Text Loader PDF loader Web Based loader CSV loader , vector store
text splitters
== recursive character text SPL semantic text splitter length based splitting chunking chunk size chunk overlap, 
retrievers 
== Web Retriever similarity multi query retrie , mmr , contextual compression , 
RAG include every details of RAG step by step implementation

tools in build-in Tools custom tools

generaet a detailed cheetsheet on these topics inlude if anyhing is missing
    """

    if not user_input:
        print("No input provided. Exiting.")
    else:
        run_cheatsheet_agent(user_input, output_file="cheatsheet_output.html")