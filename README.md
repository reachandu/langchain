# LangChain / LangGraph Learning Workspace

A personal, hands-on workspace for learning **LangChain** and **LangGraph**.

Every `.py` file here is a standalone, runnable script that demonstrates exactly one concept. There is no package, no `__init__.py`, no test suite and no build step. You run a file top-to-bottom and it prints what it did.

The folders are laid out roughly as a curriculum. Read/run them in this order:

```
llm_basics → prompts → structured_output → output_parsers → chains →
text_splitters → document_loaders → embeddings → retrievers → rag →
agents → lang_graph
```

---

## 1. Setup

### Requirements

- Python 3.14 (the checked-in `.venv` was built with `python3.14`)
- macOS / Linux shell (commands below use bash/zsh)

### Create and activate the virtual environment

The repo already ships a `.venv`. To rebuild it from scratch:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
| --- | --- |
| `langchain`, `langchain-core`, `langchain-community` | core framework |
| `langchain-google-genai` | Gemini chat + embeddings |
| `langchain-groq` | Groq-hosted Llama models |
| `langchain-huggingface`, `sentence-transformers` | local embedding models |
| `chromadb`, `langchain-chroma`, `faiss-cpu` | vector stores |
| `pypdf`, `bs4`, `wikipedia` | document loaders / sources |
| `streamlit` | UI for the one web app |
| `langgraph` | stateful agent graphs |

> **First run note:** the HuggingFace embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) is downloaded and cached on first use. That download happens once; after that embeddings run locally with no API call.

### API keys

Create a `.env` file in the repo root (one already exists here). Every script that touches an API calls `load_dotenv()` and expects:

```bash
GOOGLE_API_KEY=...      # Google AI Studio, for Gemini models
GROQ_API_KEY=...        # console.groq.com, for Llama models
HF_TOKEN=...            # HuggingFace, optional for public models
USER_AGENT=...          # any string; used by WebBaseLoader
```

> ⚠️ **Security:** `.env` is not ignored by any VCS here (this folder is not a git repo). If you ever publish this workspace, **rotate these keys first** and move them out of the file.

---

## 2. Running the scripts

**Always run from the repo root.** Data paths are hardcoded relative to the root (`sf_homeless_brief_final_web.pdf`, `docs`, `netflix_titles.csv`), so `python basic_rag.py` from inside `rag/` fails with a file-not-found error.

```bash
source .venv/bin/activate

python llm_basics/groqllama.py          # any plain script
python rag/basic_rag.py
python lang_graph/tool_calls_graph.py

streamlit run rag/db_memory_rag.py      # the only Streamlit app
```

Files written by scripts also land in the root:

| File | Written by |
| --- | --- |
| `chat_memory1.db` | `rag/db_memory_rag.py` — SQLite chat sessions |
| `leave_request_email.txt` | `lang_graph/draft_email.py` |

Interactive scripts (the RAG loops, the agents, the chat graphs) run a `while True: input()` loop and stop when you type `exit`.

> **Vector stores are in-memory only.** `Chroma.from_documents(...)` and `FAISS.from_texts(...)` are called without a `persist_directory`, so every run re-embeds the source documents from scratch. Expect a slow start on the PDF-backed scripts — that is the design, not a bug.

---

## 3. Provider conventions

Two chat providers are used interchangeably, and **their response shapes differ** — this is the most common source of breakage.

| Provider | Models | Reading the response |
| --- | --- | --- |
| **Groq** (`ChatGroq`) | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | `response.content` is a plain string |
| **Gemini** (`ChatGoogleGenerativeAI`) | `gemini-3.x-flash` | content is a list of parts → `response.content[0].get("text")` |

Groq is used for `chains/`, `structured_output/`, `output_parsers/` and the whole `rag/` folder. Gemini is used for `prompts/` and most of `lang_graph/`.

Embeddings are local HuggingFace (`all-MiniLM-L6-v2`) almost everywhere; `lang_graph/rag_agent.py` is the exception and uses `GoogleGenerativeAIEmbeddings`.

---

## 4. Folder-by-folder guide

### `llm_basics/` — "hello, model"

The smallest possible call to each provider, printing both the raw response object and the extracted text so the shape difference is visible.

| File | What it shows |
| --- | --- |
| `geminillm.py` | `ChatGoogleGenerativeAI` + `content[0].get("text")` |
| `groqllama.py` | `ChatGroq` + plain `.content` |

### `prompts/` — building the input text

| File | What it shows |
| --- | --- |
| `static_prompt.py` | `PromptTemplate` with no variables |
| `dynamic_prompt.py` | `PromptTemplate` with `{topic}` / `{style}` slots |
| `chat_prompt_template.py` | `ChatPromptTemplate` with System/Human message templates; `to_messages()` vs `to_string()` |
| `blog_post_generator.py` | Interactive: takes a topic, keeps a `chat_history` list, generates a blog post with Gemini |

### `structured_output/` — making the model return typed objects

Four ways to describe a schema to `model.with_structured_output()`, all analysing the same product review.

| File | Schema style |
| --- | --- |
| `dict_output.py` | plain `TypedDict` |
| `annotated_output.py` | `TypedDict` + `Annotated[...]` field descriptions |
| `json_schema_output.py` | a raw JSON Schema dict |
| `pydantic_output.py` | Pydantic `BaseModel` with `Field(...)` validation |

### `output_parsers/` — making sense of free-form output

| File | What it shows |
| --- | --- |
| `stroutputparser.py` | `StrOutputParser`, chained across two prompts |
| `json_output_parser.py` | `JsonOutputParser` + `get_format_instructions()` |
| `pydantic_output_parser.py` | `PydanticOutputParser` validating into a `Person` model (with age bounds) |

### `chains/` — LCEL composition (`prompt | model | parser`)

| File | What it shows |
| --- | --- |
| `simple_chain.py` | one prompt → model → parser |
| `sequential_chain.py` | report → summary; output of step 1 feeds step 2 |
| `parallel_chain.py` | `RunnableParallel`: notes + Q&A generated at once, then merged |
| `conditional_chain.py` | `RunnableBranch` / `RunnableLambda`: classify feedback sentiment, then route to a positive or negative reply prompt |

### `text_splitters/` — chunking before embedding

| File | What it shows |
| --- | --- |
| `char_text_splitter.py` | `CharacterTextSplitter` over a PDF |
| `recursive_char_splitter.py` | `RecursiveCharacterTextSplitter` over raw text, showing `chunk_size` / `chunk_overlap` |
| `python_code_splitter.py` | `PythonCodeTextSplitter` / `Language`-aware splitting of a code sample |

### `document_loaders/` — getting data into `Document`s

| File | What it shows |
| --- | --- |
| `text_loader.py` | `TextLoader` over `cats.txt` |
| `pdf_loader.py` | `PyPDFLoader` over `sample_western_life.pdf` |
| `csv_loader.py` | `CSVLoader` over `netflix_titles.csv` (large file) |
| `directory_loader.py` | `DirectoryLoader` + `lazy_load()` over every PDF in `docs/` |
| `webbase_loader.py` | `WebBaseLoader` scrapes a URL, then answers a question about the page with Groq |

### `embeddings/` — vectors and vector stores

| File | What it shows |
| --- | --- |
| `hf_embeddings.py` | `embed_query()` with a local MiniLM model, prints the vector |
| `chroma_db.py` | `Chroma.from_texts` + `similarity_search` over stock-investing sentences |
| `faiss_vs.py` | the same idea with FAISS over cricket sentences |

### `retrievers/` — retrieval strategies

| File | What it shows |
| --- | --- |
| `vs_retriever.py` | `vectorstore.as_retriever()` — the standard similarity retriever over an in-memory Chroma store |
| `mmr.py` | `search_type="mmr"` for maximal marginal relevance, i.e. diverse rather than near-duplicate results |
| `wiki_retrievers.py` | `WikipediaRetriever` — retrieval with no local store |

### `rag/` — the same pipeline, five levels of complexity

All five index `sf_homeless_brief_final_web.pdf` (`multi_doc_rag` uses `docs/` instead). When changing RAG behaviour, be clear about **which stage** you mean — the earlier files are intentionally kept simple.

| Stage | File | What it adds |
| --- | --- | --- |
| 1 | `basic_rag.py` | The reference pipeline, numbered step by step: load PDF → split → embed → vector store → retriever → prompt → llm → parser, wired as one LCEL chain. Asks a single hardcoded question. |
| 2 | `explainable_rag.py` | Retrieves manually so it can print **provenance** — which chunk and which page each piece of the answer came from. |
| 3 | `conversational_rag.py` | Adds `MessagesPlaceholder` chat history and a CLI loop, so follow-up questions have context. |
| 4 | `multi_doc_rag.py` | Indexes every PDF in `docs/`, tags each chunk with `source_document` metadata, and cites the source document per answer. |
| 5 | `db_memory_rag.py` | Streamlit app. Conversational RAG plus SQLite-backed sessions: a sidebar lists previous conversations, "New Chat" starts a fresh uuid session, and history survives restarts in `chat_memory1.db`. The document index is cached with `@st.cache_resource`. |

```bash
streamlit run rag/db_memory_rag.py
```

### `agents/` — tool calling with LangChain

| File | What it shows |
| --- | --- |
| `tool_calling_agent.py` | `create_agent()` with `@tool`-decorated `multiply`/`add` functions, a system prompt and a CLI loop. Prints the full response dict as a debug view of the agent's reasoning and tool calls. |

### `lang_graph/` — stateful graphs, roughly in order

Nodes are functions that take a `TypedDict` state, mutate it and return it. Message-based graphs use `Annotated[Sequence[BaseMessage], add_messages]`. Branching goes through a `should_continue` / `decide_next_node` function plus a string→node mapping passed to `add_conditional_edges`.

| File | What it shows |
| --- | --- |
| `basic_graph.py` | One node, `START → node → END`. The "hello world". |
| `next_graph.py` | Three nodes chained in sequence over a richer state |
| `conditional_graph.py` | `add_conditional_edges`: route to add or subtract based on an `operation` field |
| `more_conditions_graph.py` | Two independent conditional branches in one graph |
| `loop_graph.py` | A cycle: keep appending random numbers until a counter hits 5, then exit |
| `ai_agent_graph.py` | First LLM-in-a-node graph; a chat loop with Gemini |
| `tool_calls_graph.py` | `ToolNode` + `bind_tools`: the model calls `add`/`sub`/`mul` tools and loops back until it has an answer |
| `draft_email.py` | An agentic document editor: `update`/`save` tools plus a global document buffer; writes `leave_request_email.txt` in the repo root |
| `rag_agent.py` | RAG as an agent — retrieval exposed as a `@tool` the model chooses to call. The one file using `GoogleGenerativeAIEmbeddings` instead of HuggingFace. |

---

## 5. Data files in the root

| File / folder | Role |
| --- | --- |
| `sf_homeless_brief_final_web.pdf` | main RAG corpus (San Francisco housing policy brief), used by all of `rag/` and by `lang_graph/rag_agent.py` |
| `sample_western_life.pdf` | small PDF for the loader demo |
| `netflix_titles.csv` | ~3 MB CSV for `CSVLoader` |
| `cats.txt` | small text file for `TextLoader` |
| `docs/` | four topic PDFs (`finance_report`, `healthy_living`, `technology_ai`, `travel_guide`) used by `multi_doc_rag.py`, `directory_loader.py` and `char_text_splitter.py` |
| `chat_memory1.db` | *generated* — SQLite store for `db_memory_rag` |
| `leave_request_email.txt` | *generated* — output of `draft_email.py` |

---

## 6. Style conventions when adding or editing files

- Keep each script **standalone and runnable on its own**. No shared helper modules, no imports between folders.
- Some files (`rag/basic_rag.py`, `rag/explainable_rag.py`) place imports **inline at each numbered step** rather than at the top. Follow whichever style the file you are editing already uses.
- Compose with LCEL pipes (`prompt | model | parser`) and the Runnable family (`RunnableParallel`, `RunnableBranch`, `RunnableLambda`).
- Interactive scripts use a `while True` / `input()` loop that breaks on `"exit"`.
- Several files open with `warnings.filterwarnings("ignore", category=DeprecationWarning)` to silence `langchain_community` deprecation noise.
- Both `langchain_community.vectorstores.Chroma` (deprecated) and `langchain_chroma.Chroma` appear across the workspace. **Prefer `langchain_chroma` in new code.**

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError` on a PDF/CSV | You ran the script from inside its folder. Run from the repo root. |
| `'NoneType'` / `TypeError` reading `response.content` | Provider mismatch: Gemini returns a list of parts (`response.content[0].get("text")`), Groq returns a string (`response.content`). |
| Missing API key / 401 | `.env` is missing or not in the repo root, or the script was run from a different working directory so `load_dotenv()` found nothing. |
| Very slow startup on RAG scripts | Expected. The PDF is re-loaded, re-split and re-embedded on every run because no vector store is persisted. The first run also downloads the MiniLM embedding model. |
| Streamlit app shows no history | History lives in `chat_memory1.db` in the repo root; launch `streamlit` from the root so it finds the same database file. |
