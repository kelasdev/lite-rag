# Design: MCP Server (`lite-rag mcp`)

## 1. Overview

Expose the lite-rag RAG pipeline as an **MCP (Model Context Protocol) server** so that
agent clients (e.g. Hermes Agent) can query the indexed Markdown documents as tools.

The server runs as a subcommand (`lite-rag mcp`) over the **stdio transport** (JSON-RPC
2.0), spawned by the MCP client. It reuses the existing RAG pipeline
(config → DuckDB → retriever → LLM) without duplicating logic.

---

## 2. Scope

**In scope**
- New cobra subcommand `lite-rag mcp` (stdio transport)
- Three tools: `search_docs`, `rag_context`, `list_documents`
- Extraction of the shared ask/search pipeline from `cmd/lite-rag/ask.go` into `internal/rag`
- Hermes client registration documentation

**Out of scope (YAGNI)**
- HTTP/SSE MCP transport (stdio is the standard for locally spawned tools)
- Streaming tool results — MCP `tools/call` is a single request/response; results are
  returned complete (see §8)
- `index` / `reindex` tools (indexing stays a CLI concern)
- Authentication (local stdio, spawned by the client)

---

## 3. Architecture

```
 MCP Client (Hermes Agent)                    lite-rag mcp  (subprocess, stdio)
        │                                            │
        │  initialize / tools/list / tools/call       │
        │ ──────────────────────────────────────────▶ │
        │                                            │  internal/mcpserver/
        │                                            │    (go-sdk: mcp.Server)
        │                                            │      ├─ search_docs
        │                                            │      ├─ rag_context
        │                                            │      ├─ list_documents
        │                                            │      └─ index_documents
        │                                            │            │
        │                                            │    internal/rag/  (Engine)
        │                                            │      ├─ database/   (DuckDB)
        │                                            │      ├─ retriever/  (vector search)
        │                                            │      └─ llm/        (OpenAI-compatible,
        │                                            │                       SSE upstream — internal)
        │ ◀──── CallToolResult (single, complete) ─── │
```

Isolation: CLI, MCP, and any future consumer share one pipeline implementation.
Tool handlers know nothing about DuckDB or HTTP — they only see `*rag.Engine`.

---

## 4. Components

### 4.1 `cmd/lite-rag/mcp.go` — subcommand

- `lite-rag mcp` with the inherited persistent flags (`--config`, `--db`)
- Flow: load config → `database.Open` → build `*rag.Engine` → build `*mcp.Server` →
  `server.Run(ctx, &mcp.StdioTransport{})` until the client disconnects
- **stdout is the protocol channel**: all diagnostics go to stderr
  (`slog` default; `Engine.Progress` set to `io.Discard`)

### 4.2 `internal/rag` — shared pipeline (extraction from `ask.go`)

```go
// Narrow subsets of the concrete types — the Engine is testable without a
// real DuckDB connection or LLM client.
type DocDB interface {
    SimilarChunks(query []float32, topK int, embeddingModel string) ([]database.ScoredChunk, error)
    AdjacentChunks(documentID string, lo, hi int) ([]database.ChunkRow, error)
    ListDocuments() ([]database.DocumentRecord, error)
}

type LLM interface {
    Embed(ctx context.Context, texts []string) ([][]float32, error)
    Chat(ctx context.Context, messages []llm.Message, w io.Writer) error
    RewriteQuery(ctx context.Context, query string) ([]string, error)
}

type Engine struct {
    db             DocDB
    llm            LLM
    rewriter       retriever.QueryRewriter // nil = query rewriting disabled
    embeddingModel string
    retrievalCfg   config.RetrievalConfig
    Progress       io.Writer              // nil-safe; io.Discard in MCP mode
}

func NewEngine(db DocDB, client LLM, cfg *config.Config) *Engine

// Search: retrieval only. topK <= 0 uses the configured default.
func (e *Engine) Search(ctx context.Context, query string, topK int) ([]retriever.Passage, error)

// Ask: full pipeline. Tokens stream to w as they arrive AND the complete
// answer is returned (CLI passes os.Stdout; MCP passes io.Discard).
func (e *Engine) Ask(ctx context.Context, query string, w io.Writer) (string, []retriever.Passage, error)

// ListDocuments: thin pass-through to db.ListDocuments().
func (e *Engine) ListDocuments() ([]database.DocumentRecord, error)
```

The nonce-protected context block and message construction move from `runAsk`
into `Engine.Ask`. The Engine holds no mutable per-request state: a
`retriever` is constructed per call, so a `top_k` override on one request
cannot leak into the next. `cmd/lite-rag/ask.go` is refactored to call
`Engine.Ask` — **CLI behavior must not change** (streaming to stdout in text
mode, buffering in `--json` mode, identical source output).

### 4.3 `internal/mcpserver` — tool registration

- `New(engine *rag.Engine, ingester Ingester) *mcp.Server` — registers the
  four tools via the
  SDK's generic `mcp.AddTool` with typed input/output structs
- Tool `Description` fields carry the usage guidance in §5 (the agent-visible
  "when to use this" contract)
- Input/output structs use `json` + `jsonschema` tags; the SDK infers schemas

---

## 5. Tool Specifications

| Tool | Input | Output | LLM call | Purpose / trigger (agent-facing) |
|---|---|---|---|---|
| `search_docs` | `query string`, `top_k int?` | `{passages:[{content, file_path, heading_path, score}]}` | No | Semantic search over indexed Markdown. Returns raw relevant passages with file path, heading path, score. Use to locate where information lives or to read and synthesize source text yourself. |
| `rag_context` | `question string` | `{answer, sources:[{file_path, heading_path, score}]}` | Yes (buffered) | Ask a question and receive a complete answer generated strictly from document content, plus cited sources. Use for a ready-made grounded answer. Unrelated context returns a "documents do not contain information to answer this question" message (in the question's language) — not an error; related-but-indirect context is summarized with the gap noted. |
| `list_documents` | — | `{documents:[{path, chunks, indexed_at, embedding_model}]}` | No | List indexed documents. Use to check knowledge-base coverage before searching, or to answer what is indexed. |
| `index_documents` | `path string` | `{path, documents_total, documents_added}` | No (embed only) | Index a directory (recursive `*.md`, unchanged files skipped) or a single file into the knowledge base. Use after adding or changing documents so `search_docs` / `rag_context` can find them. `documents_added` counts new files; updated files refresh in place and show 0. Calls the embedding API per new chunk — can take a while on large directories. |

---

## 6. Data Flow

**`search_docs`** — `query` → `Engine.Search` → embed → `SimilarChunks` →
context expansion → passages (sorted by score) → JSON result.

**`rag_context`** — `question` → `Engine.Ask` → retrieve passages → build
nonce-tagged context block → `client.Chat` (SSE to the LLM endpoint, buffered)
→ answer + deduplicated sources → JSON result.

**`list_documents`** — `Engine.ListDocuments` → `db.ListDocuments()` → JSON result.

**`index_documents`** — `path` → `Ingester.IndexPath` (adapter over
`internal/indexer`, dispatches file vs directory) → before/after
`db.ListDocuments()` count → JSON result. Runs in the MCP process, sharing
its single DuckDB connection — no lock conflict with concurrent tool calls.

---

## 7. Error Handling

- Tool failures (DB error, LLM error) → returned as Go errors → the SDK converts
  them to `CallToolResult` with `IsError` and the message
- `search_docs` with no hits → **empty passages list**, not an error
- `index_documents` with an empty path or a path that does not exist → Go
  error → `IsError` result carrying the underlying message
- `rag_context` with no passages → a canned "No relevant documents found in
  the indexed documents." message with empty `sources`, **without calling the
  LLM** (the engine returns nil passages and `Ask` skips the chat call). The
  no-documents condition is keyed on the passage list, not the answer string:
  an empty LLM completion with passages found still returns the (empty) answer
  plus its sources rather than the canned message
- Config/DB startup failure → command exits non-zero before the transport starts

---

## 8. Streaming Clarification

MCP `tools/call` is a single request/response — there is no channel to stream
partial results to the client. Therefore:

- `rag_context` **buffers** the complete answer and returns it in one result
- The SSE streaming between lite-rag and the LLM API is unchanged — it is an
  internal upstream call, invisible to the MCP client
- `Engine.Ask(ctx, query, w)` keeps both behaviors: tokens flow to `w` as they
  arrive (CLI streaming) while the complete answer is also returned (MCP)

---

## 9. Dependencies

- `github.com/modelcontextprotocol/go-sdk` (official Go SDK, maintained with
  Google) — server construction, typed tool handlers, stdio transport
- Transitive: `github.com/google/jsonschema-go` (schema inference/validation)

Both documented in `docs/en/dependencies.md` per RULES.md Rule 18.

---

## 10. Testing

**Unit (`internal/mcpserver`, `internal/rag`)**
- SDK `InMemoryTransports`: client + server in-process; call each tool and assert
- Fakes: `DBReader`/`ListDocuments` fake, fake `Embedder`, fake `Completer`
- Coverage: each tool's happy path + error paths (DB error, empty results)

**End-to-end (`cmd/lite-rag/mcp_e2e_test.go`)**
- Temp DuckDB + local fake LLM HTTP server (OpenAI-compatible SSE)
- Spawn the real `lite-rag mcp` binary via the SDK's `CommandTransport`
- Handshake `initialize` → `tools/list` → `tools/call` for all four tools

**Regression**
- Existing CLI `ask` tests must stay green after the `internal/rag` extraction

---

## 11. Hermes Client Registration

Configure in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  lite-rag:
    command: /home/isekai/lite-rag/dist/lite-rag
    args: ["mcp", "--config", "<path to lite-rag config.toml>"]
    enabled: true
    timeout: 120        # LLM answers can be slow
```

Verification: restart the session, then call `search_docs`, `rag_context`, and
`list_documents` live against the indexed database.

---

## 12. Documentation

- This design document (`docs/en/design/mcp-server.md`) + Japanese parallel
  (`docs/ja/design/mcp-server.ja.md`) per Rule 11
- MCP usage + Hermes registration section in `docs/en/setup.md` and
  `docs/ja/setup.ja.md` per Rule 7
- Dependency entries in `docs/en/dependencies.md` per Rule 18
