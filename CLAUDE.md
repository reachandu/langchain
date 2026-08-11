# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal LangChain / LangGraph learning workspace. Each `.py` file is a standalone, runnable script demonstrating one concept — there is no package, no `__init__.py`, no test suite, no linter config, and no git repo. Scripts are executed directly, top to bottom, and print their results.

## Running

```bash
source .venv/bin/activate          # Python 3.14
pip install -r requirements.txt
python rag/basic_rag.py            # always from the repo root
streamlit run rag/db_memory_rag.py # the one Streamlit app
```

**Always run from the repo root.** Data paths are hardcoded relative to it (`PyPDFLoader("sf_homeless_brief_final_web.pdf")`, `DirectoryLoader("docs")`, `CSVLoader("netflix_titles.csv")`), so `python basic_rag.py` from inside `rag/` fails. Files written by scripts (`chat_memory1.db`, `leave_request_email.txt` from `lang_graph/draft_email.py`) also land in the root.

Every script that touches an API starts with `load_dotenv()` reading `.env` at the root: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN`, `USER_AGENT`.

## Directory map (each is a concept, ordered roughly as a curriculum)

`llm_basics` → `prompts` → `structured_output` → `output_parsers` → `chains` → `text_splitters` → `document_loaders` → `embeddings` → `retrievers` → `rag` → `agents` → `lang_graph`.

The `rag/` folder is the same pipeline built up in increasing complexity: `basic_rag` (LCEL chain) → `explainable_rag` (adds chunk/page provenance) → `conversational_rag` (adds `MessagesPlaceholder` history, CLI loop) → `multi_doc_rag` (indexes all of `docs/`, tags `source_document` metadata) → `db_memory_rag` (Streamlit UI + SQLite-backed sessions). When changing RAG behaviour, check which stage the user means — the earlier files are intentionally kept simple.

## Provider conventions

Two chat providers are used interchangeably, and **their response shapes differ** — this is the most common source of breakage:

- **Groq** (`ChatGroq`, `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) — `response.content` is a plain string. Used for chains, structured output, parsers, RAG.
- **Gemini** (`ChatGoogleGenerativeAI`, `gemini-3.x-flash`) — content comes back as a list of parts, so this code reads `response.content[0].get("text")`. Used for prompts and most of `lang_graph/`.

Embeddings are almost always local HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`, no API call, first run downloads the model); `lang_graph/rag_agent.py` is the exception, using `GoogleGenerativeAIEmbeddings`.

Vector stores are **in-memory only** — `Chroma.from_documents(...)` / `FAISS.from_texts(...)` with no `persist_directory`, so every run re-embeds from scratch. Expect slow startup on the PDF-backed scripts; that's the design, not a bug.

## Idioms to match when editing

- Imports are sometimes placed inline at each numbered step (`rag/basic_rag.py`, `rag/explainable_rag.py`) rather than at the top — follow whichever style the file already uses.
- Composition is LCEL pipes (`prompt | model | parser`) with `RunnableParallel` / `RunnableBranch` / `RunnableLambda`; `chains/` covers simple, sequential, parallel, and conditional forms.
- LangGraph nodes are `TypedDict` state in/out, mutated in place and returned; message-based graphs use `Annotated[Sequence[BaseMessage], add_messages]`. Conditional routing goes through a `should_continue`/`decide_next_node` function plus a string→node dict in `add_conditional_edges`.
- Interactive scripts use a `while True` / `input()` loop that breaks on `"exit"`.
- Several files open with `warnings.filterwarnings("ignore", category=DeprecationWarning)` to silence `langchain_community` deprecations.
- `langchain_community.vectorstores.Chroma` (deprecated) and `langchain_chroma.Chroma` are both in use across files; prefer `langchain_chroma` in new code.

## Note

`.env` is not gitignored (there is no git repo here) and contains a commented-out HuggingFace token. Worth rotating and moving out of the file if this ever gets published.
