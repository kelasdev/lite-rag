package retriever_test

import (
	"context"
	"errors"
	"testing"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
	"lite-rag/internal/retriever"
)

// hybridCfg returns a retrieval config with hybrid search explicitly enabled.
func hybridCfg(topK int, minScore float32) config.RetrievalConfig {
	return config.RetrievalConfig{
		TopK: topK, ContextWindow: 0, ChunkSize: 512, ChunkOverlap: 64,
		MinScore: minScore, Hybrid: true, CandidateK: 20,
		LexicalWeight: 0.5, LexicalMin: 0.5,
	}
}

// TestRetrieve_HybridPromotesKeywordMatch reproduces the failure that motivated
// hybrid retrieval: a document literally named "Jailbreaking-Guide" loses to
// generic security notes by a few thousandths of cosine, so with top_k=1 it
// never reaches the LLM. Keyword evidence in the path must outrank it.
func TestRetrieve_HybridPromotesKeywordMatch(t *testing.T) {
	db := openDB(t)
	insertDoc(t, db, "generic", "/notes/security.md", []database.ChunkRow{
		{ID: "g0", DocumentID: "generic", ChunkIndex: 0, Content: "exploit notes",
			Embedding: []float32{0.65, 0.76}}, // cos 0.65 with the query
	})
	insertDoc(t, db, "guide", "/guides/Jailbreaking-Guide.md", []database.ChunkRow{
		{ID: "j0", DocumentID: "guide", ChunkIndex: 0, Content: "unrelated body",
			Embedding: []float32{0.60, 0.80}}, // cos 0.60, but the path matches
	})

	emb := &mockEmbedder{vec: []float32{1, 0}}
	query := "metode jailbreak" // "metode" is a stopword; "jailbreak" is the signal

	vectorOnly, err := retriever.New(db, emb, nil, "test-model", cfgMin(1, 0, 0.5)).
		Retrieve(context.Background(), query)
	if err != nil {
		t.Fatalf("vector-only Retrieve: %v", err)
	}
	if len(vectorOnly) != 1 || vectorOnly[0].DocumentID != "generic" {
		t.Fatalf("vector-only top1 = %+v; want generic", vectorOnly)
	}

	hybrid, err := retriever.New(db, emb, nil, "test-model", hybridCfg(1, 0.5)).
		Retrieve(context.Background(), query)
	if err != nil {
		t.Fatalf("hybrid Retrieve: %v", err)
	}
	if len(hybrid) != 1 || hybrid[0].DocumentID != "guide" {
		t.Fatalf("hybrid top1 = %+v; want guide (keyword evidence must promote it)", hybrid)
	}
	if hybrid[0].Score <= 0.65 {
		t.Errorf("hybrid score = %.3f; want it boosted above the generic 0.65", hybrid[0].Score)
	}
}

// TestRetrieve_HybridAdmitsPathMatchBelowMinScore: an exact path match is
// strong enough to admit a chunk whose cosine sits under min_score, while a
// passing body mention is not.
func TestRetrieve_HybridAdmitsPathMatchBelowMinScore(t *testing.T) {
	db := openDB(t)
	insertDoc(t, db, "guide", "/guides/jailbreak.md", []database.ChunkRow{
		{ID: "g0", DocumentID: "guide", ChunkIndex: 0, Content: "no overlap",
			Embedding: []float32{0, 1}}, // cos 0
	})
	insertDoc(t, db, "mention", "/notes/misc.md", []database.ChunkRow{
		{ID: "m0", DocumentID: "mention", ChunkIndex: 0, Content: "a jailbreak is mentioned",
			Embedding: []float32{0, 1}}, // cos 0, body-only keyword (0.25 < lexical_min)
	})

	emb := &mockEmbedder{vec: []float32{1, 0}}
	got, err := retriever.New(db, emb, nil, "test-model", hybridCfg(5, 0.5)).
		Retrieve(context.Background(), "jailbreak")
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(got) != 1 || got[0].DocumentID != "guide" {
		t.Fatalf("passages = %+v; want only the path match", got)
	}
}

// TestRetrieve_HybridDisabledIgnoresKeywords keeps the opt-out honest: with
// hybrid = false the retriever behaves exactly as before, using cosine only.
func TestRetrieve_HybridDisabledIgnoresKeywords(t *testing.T) {
	db := openDB(t)
	insertDoc(t, db, "guide", "/guides/jailbreak.md", []database.ChunkRow{
		{ID: "g0", DocumentID: "guide", ChunkIndex: 0, Content: "no overlap",
			Embedding: []float32{0, 1}},
	})

	emb := &mockEmbedder{vec: []float32{1, 0}}
	got, err := retriever.New(db, emb, nil, "test-model", cfgMin(5, 0, 0.5)).
		Retrieve(context.Background(), "jailbreak")
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if got != nil {
		t.Fatalf("passages = %+v; want none (keyword search must be off)", got)
	}
}

// failingLexical simulates a keyword search failure (e.g. a database without
// the method's assumptions) while vector search keeps working.
type failingLexical struct {
	*database.DB
}

func (f failingLexical) LexicalChunks([]float32, []string, int, string) ([]database.ScoredChunk, error) {
	return nil, errors.New("lexical exploded")
}

// TestRetrieve_HybridKeywordErrorFallsBackToVector: keyword search is an
// enhancement, so its failure must degrade to vector-only, not break search.
func TestRetrieve_HybridKeywordErrorFallsBackToVector(t *testing.T) {
	db := openDB(t)
	insertDoc(t, db, "generic", "/notes/security.md", []database.ChunkRow{
		{ID: "g0", DocumentID: "generic", ChunkIndex: 0, Content: "exploit notes",
			Embedding: []float32{1, 0}},
	})

	emb := &mockEmbedder{vec: []float32{1, 0}}
	got, err := retriever.New(failingLexical{db}, emb, nil, "test-model", hybridCfg(5, 0.5)).
		Retrieve(context.Background(), "jailbreak")
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(got) != 1 || got[0].DocumentID != "generic" {
		t.Fatalf("passages = %+v; want the vector hit", got)
	}
}
