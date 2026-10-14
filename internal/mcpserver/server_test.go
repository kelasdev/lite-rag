package mcpserver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
	"lite-rag/internal/llm"
	"lite-rag/internal/rag"
)

// ── Fakes (mcpserver only sees rag.DocDB / rag.LLM) ─────────────────────────

type fakeDB struct {
	chunks    []database.ScoredChunk
	adjacent  []database.ChunkRow
	documents []database.DocumentRecord
	lastTopK  int
	simErr    error
}

func (f *fakeDB) SimilarChunks(query []float32, topK int, embeddingModel string) ([]database.ScoredChunk, error) {
	f.lastTopK = topK
	if f.simErr != nil {
		return nil, f.simErr
	}
	if topK > len(f.chunks) {
		topK = len(f.chunks)
	}
	return f.chunks[:topK], nil
}

func (f *fakeDB) AdjacentChunks(documentID string, lo, hi int) ([]database.ChunkRow, error) {
	return f.adjacent, nil
}

func (f *fakeDB) ListDocuments() ([]database.DocumentRecord, error) {
	return f.documents, nil
}

// fakeIngester stands in for the internal/indexer adapter. When db is set,
// a successful IndexPath appends one document record so the before/after
// count logic in the handler is exercised.
type fakeIngester struct {
	err   error
	paths []string
	db    *fakeDB
}

func (f *fakeIngester) IndexPath(ctx context.Context, path string) error {
	f.paths = append(f.paths, path)
	if f.err != nil {
		return f.err
	}
	if f.db != nil {
		f.db.documents = append(f.db.documents, database.DocumentRecord{
			FilePath:       path,
			TotalChunks:    1,
			IndexedAt:      time.Now().UTC().Format(time.RFC3339),
			EmbeddingModel: "test-embed",
		})
	}
	return nil
}

type fakeLLM struct {
	chatAnswer  string
	judgeAnswer string
}

func (f *fakeLLM) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1, 0, 0, 0}
	}
	return out, nil
}

func (f *fakeLLM) Chat(ctx context.Context, messages []llm.Message, w io.Writer) error {
	if strings.Contains(messages[0].Content, "relevance judge") {
		_, err := io.WriteString(w, f.judgeAnswer)
		return err
	}
	_, err := io.WriteString(w, f.chatAnswer)
	return err
}

func (f *fakeLLM) RewriteQuery(ctx context.Context, query string) ([]string, error) {
	return []string{query}, nil
}

// newTestServer wires an in-process client+server over in-memory transports.
func newTestServer(t *testing.T, db rag.DocDB, client rag.LLM) *mcp.ClientSession {
	t.Helper()
	return newTestServerWithIngester(t, db, client, &fakeIngester{})
}

func newTestServerWithIngester(t *testing.T, db rag.DocDB, client rag.LLM, ing Ingester) *mcp.ClientSession {
	t.Helper()
	cfg := &config.Config{}
	cfg.Retrieval.TopK = 3
	cfg.Retrieval.ContextWindow = 1
	cfg.Models.Embedding = "test-embed"
	engine := rag.NewEngine(db, client, cfg)
	engine.Progress = nil // treated as io.Discard

	server := New(engine, ing)
	clientSide := mcp.NewClient(&mcp.Implementation{Name: "test-client"}, nil)
	ct, st := mcp.NewInMemoryTransports()
	if _, err := server.Connect(context.Background(), st, nil); err != nil {
		t.Fatalf("server connect: %v", err)
	}
	session, err := clientSide.Connect(context.Background(), ct, nil)
	if err != nil {
		t.Fatalf("client connect: %v", err)
	}
	t.Cleanup(func() { session.Close() })
	return session
}

func toolText(t *testing.T, res *mcp.CallToolResult) string {
	t.Helper()
	if res == nil || len(res.Content) == 0 {
		return ""
	}
	tc, ok := res.Content[0].(*mcp.TextContent)
	if !ok {
		t.Fatalf("unexpected content type %T", res.Content[0])
	}
	return tc.Text
}

func indexedDB() *fakeDB {
	content := "the answer is 42"
	return &fakeDB{
		chunks: []database.ScoredChunk{
			{ChunkRow: database.ChunkRow{ID: "c1", DocumentID: "d1", ChunkIndex: 0, Content: content}, Score: 0.95, FilePath: "docs/a.md"},
		},
		adjacent: []database.ChunkRow{{ChunkIndex: 0, Content: content}},
	}
}

// ── tools/list ──────────────────────────────────────────────────────────────

func TestToolsListed(t *testing.T) {
	session := newTestServer(t, &fakeDB{}, &fakeLLM{})
	var names []string
	for tool, err := range session.Tools(context.Background(), nil) {
		if err != nil {
			t.Fatalf("list tools: %v", err)
		}
		names = append(names, tool.Name)
	}
	want := []string{"index_documents", "list_documents", "rag_context", "search_docs"}
	if len(names) != len(want) {
		t.Fatalf("want %d tools, got %d (%v)", len(want), len(names), names)
	}
	for _, w := range want {
		found := false
		for _, n := range names {
			if n == w {
				found = true
			}
		}
		if !found {
			t.Errorf("tool %q not advertised (got %v)", w, names)
		}
	}
}

// ── index_documents ─────────────────────────────────────────────────────────

// TestIndexDocuments_Directory: the ingester receives the path verbatim and
// the result reports the document-count delta (documents_added).
func TestIndexDocuments_Directory(t *testing.T) {
	db := indexedDB()
	ing := &fakeIngester{db: db}
	session := newTestServerWithIngester(t, db, &fakeLLM{}, ing)
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "index_documents",
		Arguments: map[string]any{"path": "/tmp/docs"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %s", toolText(t, res))
	}
	if len(ing.paths) != 1 || ing.paths[0] != "/tmp/docs" {
		t.Errorf("ingester paths = %v, want [/tmp/docs]", ing.paths)
	}
	var out struct {
		Path            string `json:"path"`
		DocumentsTotal  int    `json:"documents_total"`
		DocumentsAdded  int    `json:"documents_added"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if out.Path != "/tmp/docs" {
		t.Errorf("path = %q, want /tmp/docs", out.Path)
	}
	if out.DocumentsAdded != 1 || out.DocumentsTotal != 1 {
		t.Errorf("documents_added = %d, documents_total = %d; want 1 and 1",
			out.DocumentsAdded, out.DocumentsTotal)
	}
}

// TestIndexDocuments_MissingPath: an empty path is a tool error, not a
// silent no-op.
func TestIndexDocuments_MissingPath(t *testing.T) {
	session := newTestServer(t, indexedDB(), &fakeLLM{})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "index_documents",
		Arguments: map[string]any{"path": ""},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("empty path must be an error result, got: %s", toolText(t, res))
	}
	if !strings.Contains(toolText(t, res), "path is required") {
		t.Errorf("error message = %q, want it to mention path is required", toolText(t, res))
	}
}

// TestIndexDocuments_Error: an indexing failure surfaces as an IsError
// result with the underlying message.
func TestIndexDocuments_Error(t *testing.T) {
	ing := &fakeIngester{err: errors.New("embed failed")}
	session := newTestServerWithIngester(t, indexedDB(), &fakeLLM{}, ing)
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "index_documents",
		Arguments: map[string]any{"path": "/tmp/docs"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("ingest failure must be an error result, got: %s", toolText(t, res))
	}
	if !strings.Contains(toolText(t, res), "embed failed") {
		t.Errorf("error message = %q, want it to carry the underlying error", toolText(t, res))
	}
}

// ── search_docs ─────────────────────────────────────────────────────────────

func TestSearchDocs(t *testing.T) {
	session := newTestServer(t, indexedDB(), &fakeLLM{})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "what is the answer?"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %s", toolText(t, res))
	}
	var out struct {
		Passages []struct {
			Content  string  `json:"content"`
			FilePath string  `json:"file_path"`
			Score    float32 `json:"score"`
		} `json:"passages"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if len(out.Passages) != 1 || out.Passages[0].FilePath != "docs/a.md" {
		t.Fatalf("unexpected passages: %+v", out.Passages)
	}
	if !strings.Contains(out.Passages[0].Content, "the answer is 42") {
		t.Errorf("passage content wrong: %q", out.Passages[0].Content)
	}
}

func TestSearchDocs_TopKOverride(t *testing.T) {
	db := indexedDB()
	session := newTestServer(t, db, &fakeLLM{})
	_, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "q", "top_k": 1},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if db.lastTopK != 1 {
		t.Errorf("want SimilarChunks called with topK 1, got %d", db.lastTopK)
	}
}

func TestSearchDocs_EmptyIsNotError(t *testing.T) {
	session := newTestServer(t, &fakeDB{}, &fakeLLM{})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "nothing matches"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("empty result must not be an error: %s", toolText(t, res))
	}
	var out struct {
		Passages []any `json:"passages"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(out.Passages) != 0 {
		t.Errorf("want 0 passages, got %d", len(out.Passages))
	}
}

// ── rag_context ─────────────────────────────────────────────────────────────

func TestRagContext(t *testing.T) {
	session := newTestServer(t, indexedDB(), &fakeLLM{chatAnswer: "The answer is 42."})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "rag_context",
		Arguments: map[string]any{"question": "what is the answer?"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %s", toolText(t, res))
	}
	var out struct {
		Answer  string `json:"answer"`
		Sources []struct {
			FilePath string  `json:"file_path"`
			Score    float32 `json:"score"`
		} `json:"sources"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if out.Answer != "The answer is 42." {
		t.Errorf("want answer %q, got %q", "The answer is 42.", out.Answer)
	}
	if len(out.Sources) != 1 || out.Sources[0].FilePath != "docs/a.md" {
		t.Fatalf("unexpected sources: %+v", out.Sources)
	}
}

func TestRagContext_NoDocuments(t *testing.T) {
	session := newTestServer(t, &fakeDB{}, &fakeLLM{chatAnswer: "must not be called"})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "rag_context",
		Arguments: map[string]any{"question": "anything"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	var out struct {
		Answer  string `json:"answer"`
		Sources []any  `json:"sources"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.Answer != "No relevant documents found in the indexed documents." {
		t.Errorf("want explicit no-documents message, got %q", out.Answer)
	}
	if len(out.Sources) != 0 {
		t.Errorf("want 0 sources, got %d", len(out.Sources))
	}
}

func TestRagContext_EmptyAnswerWithSources(t *testing.T) {
	// A provider whose completion yields no content (usage-only finish,
	// refusal) must NOT be reported as "no documents" while passages —
	// and therefore sources — exist.
	session := newTestServer(t, indexedDB(), &fakeLLM{chatAnswer: ""})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "rag_context",
		Arguments: map[string]any{"question": "what is the answer?"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %s", toolText(t, res))
	}
	var out struct {
		Answer  string `json:"answer"`
		Sources []struct {
			FilePath string `json:"file_path"`
		} `json:"sources"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if out.Answer == "No relevant documents found in the indexed documents." {
		t.Error("empty LLM completion with passages found must not claim 'no documents'")
	}
	if len(out.Sources) != 1 || out.Sources[0].FilePath != "docs/a.md" {
		t.Errorf("sources must be returned when passages were found, got %+v", out.Sources)
	}
}

// TestRagContext_JudgeRejectsKeepsEngineFallback: when the engine's judge
// rejects (translated answer set, passages suppressed), the handler must not
// overwrite the answer with the canned English message, and sources must
// stay empty — a no-answer must never carry citations.
func TestRagContext_JudgeRejectsKeepsEngineFallback(t *testing.T) {
	session := newTestServer(t, indexedDB(), &fakeLLM{
		judgeAnswer: "NO Tidak ada dalam dokumen.",
		chatAnswer:  "must not be generated",
	})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "rag_context",
		Arguments: map[string]any{"question": "siapakah ASEP"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %s", toolText(t, res))
	}
	var out struct {
		Answer  string `json:"answer"`
		Sources []struct {
			FilePath string `json:"file_path"`
		} `json:"sources"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if out.Answer != "Tidak ada dalam dokumen." {
		t.Errorf("answer = %q, want the engine's translated fallback", out.Answer)
	}
	if len(out.Sources) != 0 {
		t.Errorf("sources must be empty on a no-answer, got %+v", out.Sources)
	}
}

func TestRagContext_UnicodeAnswer(t *testing.T) {
	session := newTestServer(t, indexedDB(), &fakeLLM{chatAnswer: "日本語の回答です。"})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "rag_context",
		Arguments: map[string]any{"question": "質問"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	var out struct {
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if out.Answer != "日本語の回答です。" {
		t.Errorf("unicode answer mangled: %q", out.Answer)
	}
}

// ── list_documents ──────────────────────────────────────────────────────────

func TestListDocuments(t *testing.T) {
	db := &fakeDB{documents: []database.DocumentRecord{
		{FilePath: "docs/a.md", TotalChunks: 5, IndexedAt: "2026-10-14 10:00:00", EmbeddingModel: "test-embed"},
	}}
	session := newTestServer(t, db, &fakeLLM{})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "list_documents",
		Arguments: map[string]any{},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	var out struct {
		Documents []struct {
			Path           string `json:"path"`
			Chunks         int    `json:"chunks"`
			IndexedAt      string `json:"indexed_at"`
			EmbeddingModel string `json:"embedding_model"`
		} `json:"documents"`
	}
	if err := json.Unmarshal([]byte(toolText(t, res)), &out); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, toolText(t, res))
	}
	if len(out.Documents) != 1 {
		t.Fatalf("want 1 document, got %d", len(out.Documents))
	}
	d := out.Documents[0]
	if d.Path != "docs/a.md" || d.Chunks != 5 || d.EmbeddingModel != "test-embed" {
		t.Errorf("unexpected document: %+v", d)
	}
}

// ── error surfacing ─────────────────────────────────────────────────────────

func TestToolErrorIsSurfaced(t *testing.T) {
	db := &fakeDB{simErr: errors.New("db down")}
	session := newTestServer(t, db, &fakeLLM{})
	res, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "q"},
	})
	if err == nil && (res == nil || !res.IsError) {
		t.Fatalf("want failure surfaced, got res=%+v err=%v", res, err)
	}
	if res != nil && !strings.Contains(toolText(t, res), "db down") {
		t.Errorf("error message must contain the cause, got %q", toolText(t, res))
	}
}
