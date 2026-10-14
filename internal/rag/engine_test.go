package rag

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
	"lite-rag/internal/llm"
	"lite-rag/internal/retriever"
)

// ── Fakes ───────────────────────────────────────────────────────────────────

type fakeDB struct {
	chunks    []database.ScoredChunk
	adjacent  []database.ChunkRow
	documents []database.DocumentRecord
	lastTopK  int
	listErr   error
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
	if f.listErr != nil {
		return nil, f.listErr
	}
	return f.documents, nil
}

type fakeLLM struct {
	chatAnswer string
	chatErr    error
	messages   []llm.Message // captured on last Chat call
}

func (f *fakeLLM) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1, 0, 0, 0}
	}
	return out, nil
}

func (f *fakeLLM) Chat(ctx context.Context, messages []llm.Message, w io.Writer) error {
	f.messages = messages
	if f.chatErr != nil {
		return f.chatErr
	}
	_, err := io.WriteString(w, f.chatAnswer)
	return err
}

func (f *fakeLLM) RewriteQuery(ctx context.Context, query string) ([]string, error) {
	return []string{query}, nil
}

func testEngine(db DocDB, client LLM) *Engine {
	cfg := &config.Config{}
	cfg.Retrieval.TopK = 3
	cfg.Retrieval.ContextWindow = 1
	cfg.Models.Embedding = "test-embed"
	e := NewEngine(db, client, cfg)
	e.Progress = io.Discard
	return e
}

// ── Search ──────────────────────────────────────────────────────────────────

func TestSearch_ReturnsPassages(t *testing.T) {
	db := &fakeDB{
		chunks: []database.ScoredChunk{
			{ChunkRow: database.ChunkRow{ID: "c1", DocumentID: "d1", ChunkIndex: 0, Content: "chunk zero"}, Score: 0.9, FilePath: "docs/a.md"},
		},
		adjacent: []database.ChunkRow{
			{ChunkIndex: 0, Content: "chunk zero"},
			{ChunkIndex: 1, Content: "chunk one"},
		},
	}
	passages, err := testEngine(db, &fakeLLM{}).Search(context.Background(), "question", 0)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(passages) != 1 {
		t.Fatalf("want 1 passage, got %d", len(passages))
	}
	if passages[0].FilePath != "docs/a.md" {
		t.Errorf("want file path docs/a.md, got %q", passages[0].FilePath)
	}
	if !strings.Contains(passages[0].Content, "chunk one") {
		t.Errorf("expanded passage should contain adjacent chunk, got %q", passages[0].Content)
	}
}

func TestSearch_DefaultTopKFromConfig(t *testing.T) {
	db := &fakeDB{
		chunks:   []database.ScoredChunk{{ChunkRow: database.ChunkRow{ID: "c1", DocumentID: "d1", ChunkIndex: 0}, Score: 0.9, FilePath: "a.md"}},
		adjacent: []database.ChunkRow{{ChunkIndex: 0, Content: "x"}},
	}
	if _, err := testEngine(db, &fakeLLM{}).Search(context.Background(), "q", 0); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if db.lastTopK != 3 {
		t.Errorf("topK=0 must fall back to configured default 3, got %d", db.lastTopK)
	}
}

func TestSearch_TopKOverride(t *testing.T) {
	db := &fakeDB{
		chunks: []database.ScoredChunk{
			{ChunkRow: database.ChunkRow{ID: "c1", DocumentID: "d1", ChunkIndex: 0}, Score: 0.9, FilePath: "a.md"},
			{ChunkRow: database.ChunkRow{ID: "c2", DocumentID: "d1", ChunkIndex: 1}, Score: 0.8, FilePath: "a.md"},
			{ChunkRow: database.ChunkRow{ID: "c3", DocumentID: "d1", ChunkIndex: 2}, Score: 0.7, FilePath: "a.md"},
		},
		adjacent: []database.ChunkRow{{ChunkIndex: 0, Content: "x"}},
	}
	if _, err := testEngine(db, &fakeLLM{}).Search(context.Background(), "q", 1); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if db.lastTopK != 1 {
		t.Errorf("want SimilarChunks called with topK 1, got %d", db.lastTopK)
	}
}

func TestSearch_EmptyIsNotError(t *testing.T) {
	passages, err := testEngine(&fakeDB{}, &fakeLLM{}).Search(context.Background(), "q", 0)
	if err != nil {
		t.Fatalf("Search on empty DB must not error: %v", err)
	}
	if len(passages) != 0 {
		t.Errorf("want 0 passages, got %d", len(passages))
	}
}

func TestSearch_DBError(t *testing.T) {
	db := &fakeDB{simErr: errors.New("db down")}
	if _, err := testEngine(db, &fakeLLM{}).Search(context.Background(), "q", 0); err == nil {
		t.Fatal("want error when SimilarChunks fails")
	}
}

// ── ListDocuments ───────────────────────────────────────────────────────────

func TestListDocuments(t *testing.T) {
	db := &fakeDB{documents: []database.DocumentRecord{
		{FilePath: "docs/a.md", TotalChunks: 3, IndexedAt: "2026-10-14 10:00:00", EmbeddingModel: "test-embed"},
	}}
	docs, err := testEngine(db, &fakeLLM{}).ListDocuments()
	if err != nil {
		t.Fatalf("ListDocuments: %v", err)
	}
	if len(docs) != 1 || docs[0].FilePath != "docs/a.md" {
		t.Fatalf("unexpected documents: %+v", docs)
	}
}

func TestListDocuments_Error(t *testing.T) {
	db := &fakeDB{listErr: errors.New("db down")}
	if _, err := testEngine(db, &fakeLLM{}).ListDocuments(); err == nil {
		t.Fatal("want error when ListDocuments fails")
	}
}

// ── DedupeSources ───────────────────────────────────────────────────────────

func TestDedupeSources(t *testing.T) {
	passages := []retriever.Passage{
		{FilePath: "a.md", HeadingPath: "H1", Score: 0.9},
		{FilePath: "b.md", HeadingPath: "H2", Score: 0.8},
		{FilePath: "a.md", HeadingPath: "H3", Score: 0.7}, // lower score, must not replace
	}
	sources := DedupeSources(passages)
	if len(sources) != 2 {
		t.Fatalf("want 2 sources, got %d", len(sources))
	}
	if sources[0].FilePath != "a.md" || sources[0].Score != 0.9 || sources[0].HeadingPath != "H1" {
		t.Errorf("first source wrong: %+v", sources[0])
	}
	if sources[1].FilePath != "b.md" {
		t.Errorf("order must follow first occurrence, got %+v", sources[1])
	}
}

func TestDedupeSources_Unicode(t *testing.T) {
	passages := []retriever.Passage{
		{FilePath: "日本語/ドキュメント.md", HeadingPath: "見出し", Score: 0.5},
	}
	sources := DedupeSources(passages)
	if len(sources) != 1 {
		t.Fatalf("want 1 source, got %d", len(sources))
	}
	if sources[0].FilePath != "日本語/ドキュメント.md" || sources[0].HeadingPath != "見出し" {
		t.Errorf("unicode mangled: %+v", sources[0])
	}
}
