# Setup Guide

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Go | 1.26.9+ | Build toolchain |
| make | any | Build automation |
| cc (clang/gcc) | any | CGo compilation (required by go-duckdb) |
| podman or docker | any | Linux cross-compilation (optional) |

> **Note:** The build uses a project-local Go module cache (`.go/`) to avoid
> polluting `~/go`. This is configured automatically by the Makefile.

## Quick Start

```sh
git clone <repo-url> lite-rag
cd lite-rag

# Copy and edit configuration
mkdir -p ~/.config/lite-rag
cp config.example.toml ~/.config/lite-rag/config.toml
$EDITOR ~/.config/lite-rag/config.toml

# Install git hooks and build tools
make setup

# Build
make build

# Run
./dist/lite-rag --help
```

## LM Studio Configuration

1. Start LM Studio and load the following models:
   - Embedding: `nomic-ai/nomic-embed-text-v1.5-GGUF`
   - Chat: `openai/gpt-oss-20b`
2. Enable the local server (default port: `1234`).
3. Confirm the API is reachable:
   ```sh
   curl http://localhost:1234/v1/models
   ```

## Security Scanning

`govulncheck` scans Go dependencies for known vulnerabilities and runs as part of `make check`.

```sh
# Install manually (also done by make setup)
go install golang.org/x/vuln/cmd/govulncheck@latest

# Run standalone
make vuln
```

## Linting

`make lint` runs [golangci-lint](https://golangci-lint.run) v2. The tool is
installed into `.go/bin/` by `make setup` (or on first `make lint`).

Lint behavior is pinned by `.golangci.yml`. v2 enables no exclusion presets
by default, so the file explicitly enables the presets v1 applied
automatically (`comments`, `std-error-handling`, `common-false-positives`,
`legacy`) — the lint gate behaves the same as before the upgrade.

## MCP Server

`lite-rag` can run as an MCP (Model Context Protocol) server so AI agents can
query the indexed documents as tools:

```sh
lite-rag mcp
```

Pass `--web` to also serve the Web UI from the same process (recommended —
one process, one DuckDB connection, no lock conflict):

```sh
lite-rag mcp --web                      # uses server.addr from the config
lite-rag mcp --web --web-addr :8080     # or override the listen address
```

The server speaks JSON-RPC 2.0 over stdin/stdout (the stdio transport); all
diagnostics go to stderr. It exposes three tools:

| Tool | Purpose |
|------|---------|
| `search_docs` | Semantic search returning raw passages (content, file path, heading path, score) — no LLM call |
| `rag_context` | Full Q&A: document-grounded answer plus cited sources |
| `list_documents` | List indexed documents (path, chunks, indexed timestamp, embedding model) |

### Registering with an MCP client

Add the server to the client's configuration. Example for Hermes Agent
(`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  lite-rag:
    command: /home/isekai/lite-rag/dist/lite-rag
    args: ["mcp", "--web", "--config", "/home/isekai/.config/lite-rag/config.toml"]
    enabled: true
    timeout: 120
```

`timeout: 120` is recommended — `rag_context` calls the LLM and answers can
take a while. The `--config` path must point at the same config (and therefore
the same DuckDB database) that was indexed.

## Git Hook Installation

Git hooks are managed under `scripts/hooks/`. To install them:

```sh
make setup
```

This copies `scripts/hooks/pre-commit` and `scripts/hooks/pre-push` to `.git/hooks/`
and makes them executable. Both hooks run `make check` (lint + test + build) before
allowing the operation.

## Environment Variables

All settings in the config file can be overridden via environment variables:

| Variable | Overrides |
|---|---|
| `LITE_RAG_API_BASE_URL` | `api.base_url` |
| `LITE_RAG_API_KEY` | `api.api_key` |
| `LITE_RAG_API_EMBED_URL` | `api.embed_url` |
| `LITE_RAG_API_EMBED_API_KEY` | `api.embed_api_key` |
| `LITE_RAG_EMBEDDING_MODEL` | `models.embedding` |
| `LITE_RAG_CHAT_MODEL` | `models.chat` |
| `LITE_RAG_DB_PATH` | `database.path` |

### Separate Chat and Embedding Servers

Chat completions and embeddings can be served by two different servers. Set
`api.embed_url` and `api.embed_api_key` to route `/embeddings` elsewhere:

```toml
[api]
base_url       = "http://localhost:1234/v1"  # /chat/completions
api_key        = "lm-studio"
embed_url      = "http://localhost:8000/v1"  # /embeddings
embed_api_key  = "embed-key"
```

Leaving `embed_url` / `embed_api_key` empty falls back to `base_url` /
`api_key`, so existing configs keep working unchanged.

The default config file path follows the XDG Base Directory Specification:
`$XDG_CONFIG_HOME/lite-rag/config.toml` (defaults to `~/.config/lite-rag/config.toml`).
Override with the `--config` flag: `lite-rag --config /path/to/config.toml`.

## Cross-Compilation

### darwin targets (from a macOS host)

Uses the macOS system `clang` with `-arch` for multi-architecture support.
No additional tools required.

```sh
make cross-build-darwin
# Produces: dist/lite-rag-darwin-arm64
```

### linux targets (via container)

`go-duckdb` links a prebuilt static DuckDB library built with GCC/libstdc++.
Cross-compiling this from macOS requires a Linux environment with the matching
C++ runtime. The easiest approach is to run the build inside a container:

```sh
# Requires podman (brew install podman) or docker
make cross-build-linux
# Produces: dist/lite-rag-linux-amd64  dist/lite-rag-linux-arm64
```

The Makefile detects `podman` first, then falls back to `docker`.

### linux targets (on a Linux host)

If you are already on a Linux machine, build directly without a container:

```sh
# For arm64, install the cross-compiler first:
sudo apt-get install gcc-aarch64-linux-gnu

make cross-build-linux-native
```


## Project-Local Module Cache

The Makefile sets:

```
GOPATH=$(pwd)/.go
GOMODCACHE=$(pwd)/.go/pkg/mod
GOCACHE=$(pwd)/.go/cache
```

This keeps all downloaded modules and build artifacts inside the project
directory, which is useful in restricted environments. The `.go/` directory
is git-ignored.
