// Package mcpserver exposes the lite-rag RAG pipeline as MCP tools.
package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"lite-rag/internal/rag"
)

// Ingester indexes Markdown documents into the database. cmd/lite-rag/mcp.go
// supplies an adapter over internal/indexer that dispatches file vs directory.
type Ingester interface {
	IndexPath(ctx context.Context, path string) error
}

// New builds an MCP server exposing search_docs, rag_context,
// list_documents, and index_documents over the engine's pipeline.
func New(engine *rag.Engine, ingester Ingester) *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{Name: "lite-rag"}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name: "search_docs",
		Description: "Semantic search over indexed Markdown documents. Returns the most " +
			"relevant text passages with file path, heading path, and similarity score. " +
			"Does NOT call the LLM — you receive raw source text. Use to locate where " +
			"information lives, or to read and synthesize passages yourself.",
	}, searchDocs(engine))

	mcp.AddTool(server, &mcp.Tool{
		Name: "rag_context",
		Description: "Ask a question against the indexed documents and receive a complete " +
			"answer generated strictly from document content, plus cited sources with " +
			"scores. Calls the LLM internally; the answer is buffered and returned as a " +
			"single result. Use for a ready-made grounded answer. Low relevance returns " +
			"a 'not contained in documents' message — not an error.",
	}, ragContext(engine))

	mcp.AddTool(server, &mcp.Tool{
		Name: "list_documents",
		Description: "List every document currently indexed in the lite-rag database: " +
			"file path, chunk count, indexed timestamp, embedding model. No LLM call. " +
			"Use to check knowledge-base coverage before searching, or to answer what " +
			"is indexed.",
	}, listDocuments(engine))

	mcp.AddTool(server, &mcp.Tool{
		Name: "index_documents",
		Description: "Index Markdown documents into the knowledge base. Pass a directory " +
			"(recursively indexes *.md, skipping unchanged files) or a single file path. " +
			"Use after adding or changing documents so search_docs / rag_context can " +
			"find them. Calls the embedding API for every new chunk — on large " +
			"directories this can take a while.",
	}, indexDocuments(engine, ingester))

	return server
}

// ── search_docs ─────────────────────────────────────────────────────────────

type searchIn struct {
	Query string `json:"query" jsonschema:"the search query"`
	TopK  int    `json:"top_k,omitempty" jsonschema:"maximum number of passages to return; 0 or omitted uses the configured default"`
}

type passageJSON struct {
	Content     string  `json:"content"`
	FilePath    string  `json:"file_path"`
	HeadingPath string  `json:"heading_path,omitempty"`
	Score       float32 `json:"score"`
}

type searchOut struct {
	Passages []passageJSON `json:"passages"`
}

func searchDocs(engine *rag.Engine) func(context.Context, *mcp.CallToolRequest, searchIn) (*mcp.CallToolResult, any, error) {
	return func(ctx context.Context, _ *mcp.CallToolRequest, in searchIn) (*mcp.CallToolResult, any, error) {
		passages, err := engine.Search(ctx, in.Query, in.TopK)
		if err != nil {
			return nil, nil, fmt.Errorf("search_docs: %w", err)
		}
		out := searchOut{Passages: make([]passageJSON, 0, len(passages))}
		for _, p := range passages {
			out.Passages = append(out.Passages, passageJSON{
				Content:     p.Content,
				FilePath:    p.FilePath,
				HeadingPath: p.HeadingPath,
				Score:       p.Score,
			})
		}
		return textResult(out)
	}
}

// ── rag_context ─────────────────────────────────────────────────────────────

type askIn struct {
	Question string `json:"question" jsonschema:"the question to answer from the indexed documents"`
}

type sourceJSON struct {
	FilePath    string  `json:"file_path"`
	HeadingPath string  `json:"heading_path,omitempty"`
	Score       float32 `json:"score"`
}

type askOut struct {
	Answer  string       `json:"answer"`
	Sources []sourceJSON `json:"sources"`
}

func ragContext(engine *rag.Engine) func(context.Context, *mcp.CallToolRequest, askIn) (*mcp.CallToolResult, any, error) {
	return func(ctx context.Context, _ *mcp.CallToolRequest, in askIn) (*mcp.CallToolResult, any, error) {
		// io.Discard: MCP tool results are single-shot; the engine buffers
		// the answer and returns it in the result below.
		answer, passages, err := engine.Ask(ctx, in.Question, io.Discard)
		if err != nil {
			return nil, nil, fmt.Errorf("rag_context: %w", err)
		}
		// An empty answer with passages found (usage-only completion,
		// refusal) is not "no documents" — only a nil passage list is.
		// Branching on the answer string would misreport it while still
		// citing sources. A judge rejection returns the engine's own
		// translated fallback with passages already suppressed, so the
		// canned message applies only when the LLM was never called.
		if answer == "" && len(passages) == 0 {
			answer = "No relevant documents found in the indexed documents."
		}
		out := askOut{Answer: answer, Sources: make([]sourceJSON, 0)}
		for _, s := range rag.DedupeSources(passages) {
			out.Sources = append(out.Sources, sourceJSON{
				FilePath:    s.FilePath,
				HeadingPath: s.HeadingPath,
				Score:       s.Score,
			})
		}
		return textResult(out)
	}
}

// ── list_documents ──────────────────────────────────────────────────────────

type docJSON struct {
	Path           string `json:"path"`
	Chunks         int    `json:"chunks"`
	IndexedAt      string `json:"indexed_at"`
	EmbeddingModel string `json:"embedding_model"`
}

type listOut struct {
	Documents []docJSON `json:"documents"`
}

func listDocuments(engine *rag.Engine) func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, any, error) {
	return func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		docs, err := engine.ListDocuments()
		if err != nil {
			return nil, nil, fmt.Errorf("list_documents: %w", err)
		}
		out := listOut{Documents: make([]docJSON, 0, len(docs))}
		for _, d := range docs {
			out.Documents = append(out.Documents, docJSON{
				Path:           d.FilePath,
				Chunks:         d.TotalChunks,
				IndexedAt:      d.IndexedAt,
				EmbeddingModel: d.EmbeddingModel,
			})
		}
		return textResult(out)
	}
}

// ── index_documents ─────────────────────────────────────────────────────────

type indexIn struct {
	Path string `json:"path" jsonschema:"directory to index recursively (*.md) or a single Markdown file path"`
}

type indexOut struct {
	Path           string `json:"path"`
	DocumentsTotal int    `json:"documents_total"`
	// DocumentsAdded counts newly-indexed files; re-indexing an updated
	// file refreshes it in place and shows as 0 added.
	DocumentsAdded int `json:"documents_added"`
}

func indexDocuments(engine *rag.Engine, ingester Ingester) func(context.Context, *mcp.CallToolRequest, indexIn) (*mcp.CallToolResult, any, error) {
	return func(ctx context.Context, _ *mcp.CallToolRequest, in indexIn) (*mcp.CallToolResult, any, error) {
		if strings.TrimSpace(in.Path) == "" {
			return nil, nil, fmt.Errorf("index_documents: path is required")
		}
		before, err := engine.ListDocuments()
		if err != nil {
			return nil, nil, fmt.Errorf("index_documents: %w", err)
		}
		if err := ingester.IndexPath(ctx, in.Path); err != nil {
			return nil, nil, fmt.Errorf("index_documents: %w", err)
		}
		after, err := engine.ListDocuments()
		if err != nil {
			return nil, nil, fmt.Errorf("index_documents: %w", err)
		}
		out := indexOut{
			Path:           in.Path,
			DocumentsTotal: len(after),
			DocumentsAdded: len(after) - len(before),
		}
		return textResult(out)
	}
}

// textResult wraps a JSON-serializable payload in a TextContent result.
func textResult(v any) (*mcp.CallToolResult, any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal result: %w", err)
	}
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: string(b)}},
	}, nil, nil
}
