// Package rag wires the retrieval-augmented generation pipeline shared by the
// CLI (ask) and the MCP server (rag_context / search_docs tools).
package rag

import (
	"context"
	"io"
	"os"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
	"lite-rag/internal/llm"
	"lite-rag/internal/retriever"
)

// DocDB is the subset of database.DB used by the Engine.
type DocDB interface {
	SimilarChunks(query []float32, topK int, embeddingModel string) ([]database.ScoredChunk, error)
	AdjacentChunks(documentID string, lo, hi int) ([]database.ChunkRow, error)
	ListDocuments() ([]database.DocumentRecord, error)
}

// LLM is the subset of llm.Client used by the Engine.
type LLM interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Chat(ctx context.Context, messages []llm.Message, w io.Writer) error
	RewriteQuery(ctx context.Context, query string) ([]string, error)
}

// Source is a deduplicated citation: one entry per source file.
type Source struct {
	FilePath    string
	HeadingPath string
	Score       float32
}

// Engine executes the search and ask stages of the RAG pipeline.
type Engine struct {
	db             DocDB
	llm            LLM
	rewriter       retriever.QueryRewriter // nil = query rewriting disabled
	embeddingModel string
	retrievalCfg   config.RetrievalConfig
	Progress       io.Writer // progress/diagnostic output; nil-safe (treated as io.Discard)
}

// NewEngine builds an Engine from config. db and client must be non-nil.
func NewEngine(db DocDB, client LLM, cfg *config.Config) *Engine {
	var rewriter retriever.QueryRewriter
	if cfg.Retrieval.QueryRewrite {
		rewriter = client
	}
	return &Engine{
		db:             db,
		llm:            client,
		rewriter:       rewriter,
		embeddingModel: cfg.Models.Embedding,
		retrievalCfg:   cfg.Retrieval,
		Progress:       os.Stderr,
	}
}

// retriever builds a Retriever for a single call. topK > 0 overrides the
// configured top_k; 0 uses the configured value. Construction is stateless,
// so a per-call retriever keeps top_k overrides free of shared state.
func (e *Engine) retriever(topK int) *retriever.Retriever {
	rcfg := e.retrievalCfg
	if topK > 0 {
		rcfg.TopK = topK
	}
	return retriever.New(e.db, e.llm, e.rewriter, e.embeddingModel, rcfg)
}

// Search retrieves passages for query without calling the chat model.
// topK <= 0 uses the configured default.
func (e *Engine) Search(ctx context.Context, query string, topK int) ([]retriever.Passage, error) {
	return e.retriever(topK).Retrieve(ctx, query)
}

// ListDocuments returns all indexed documents.
func (e *Engine) ListDocuments() ([]database.DocumentRecord, error) {
	return e.db.ListDocuments()
}

// DedupeSources collapses passages by file path, keeping the highest-scoring
// passage per file, in first-occurrence order.
func DedupeSources(passages []retriever.Passage) []Source {
	type entry struct {
		headingPath string
		score       float32
	}
	seen := make(map[string]entry, len(passages))
	order := make([]string, 0, len(passages))
	for _, p := range passages {
		if prev, ok := seen[p.FilePath]; !ok {
			order = append(order, p.FilePath)
			seen[p.FilePath] = entry{p.HeadingPath, p.Score}
		} else if p.Score > prev.score {
			seen[p.FilePath] = entry{p.HeadingPath, p.Score}
		}
	}
	sources := make([]Source, 0, len(order))
	for _, f := range order {
		en := seen[f]
		sources = append(sources, Source{FilePath: f, HeadingPath: en.headingPath, Score: en.score})
	}
	return sources
}
