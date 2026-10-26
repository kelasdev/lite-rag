package database_test

import (
	"testing"
	"time"

	"lite-rag/internal/database"
)

// insertLexDoc stores a one-chunk document under the given embedding model.
func insertLexDoc(t *testing.T, db *database.DB, id, filePath, model string, chunk database.ChunkRow) {
	t.Helper()
	chunk.ID = id + "-c0"
	chunk.DocumentID = id
	chunk.ChunkIndex = 0
	doc := database.DocumentRow{
		ID: id, FilePath: filePath, FileHash: "h-" + id,
		TotalChunks: 1, IndexedAt: time.Now().UTC(), EmbeddingModel: model,
	}
	if err := db.ReplaceDocument(doc, []database.ChunkRow{chunk}); err != nil {
		t.Fatalf("ReplaceDocument(%s): %v", id, err)
	}
}

// TestLexicalChunks_ScoresPathHeadingAndContent proves the keyword signal is
// weighted by where the term appears: a document named after the query beats
// one that only mentions the term in a heading, which beats a passing mention
// in the body.
func TestLexicalChunks_ScoresPathHeadingAndContent(t *testing.T) {
	db := newTestDB(t)
	insertLexDoc(t, db, "path", "/guides/Jailbreaking-Guide.md", "m",
		database.ChunkRow{Content: "alpha"})
	insertLexDoc(t, db, "heading", "/notes/other.md", "m",
		database.ChunkRow{HeadingPath: "# Jailbreak basics", Content: "beta"})
	insertLexDoc(t, db, "body", "/notes/misc.md", "m",
		database.ChunkRow{Content: "a jailbreak appears here"})

	hits, err := db.LexicalChunks(nil, []string{"jailbreak"}, 10, "m")
	if err != nil {
		t.Fatalf("LexicalChunks: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("hits = %d; want 3 (%+v)", len(hits), hits)
	}
	wantScores := []float32{0.8, 0.6, 0.25}
	wantDocs := []string{"path", "heading", "body"}
	for i := range wantScores {
		if hits[i].DocumentID != wantDocs[i] {
			t.Errorf("hit %d = %s; want %s (order: path > heading > body)", i, hits[i].DocumentID, wantDocs[i])
		}
		if diff := hits[i].Score - wantScores[i]; diff > 0.001 || diff < -0.001 {
			t.Errorf("hit %d score = %.3f; want %.2f", i, hits[i].Score, wantScores[i])
		}
	}
}

// TestLexicalChunks_FiltersByEmbeddingModel keeps keyword results inside the
// vector space that produced them: a chunk embedded with another model cannot
// be compared with this query.
func TestLexicalChunks_FiltersByEmbeddingModel(t *testing.T) {
	db := newTestDB(t)
	insertLexDoc(t, db, "current", "/guides/jailbreak.md", "m",
		database.ChunkRow{Content: "alpha"})
	insertLexDoc(t, db, "stale", "/guides/jailbreak-old.md", "other-model",
		database.ChunkRow{Content: "beta"})

	hits, err := db.LexicalChunks(nil, []string{"jailbreak"}, 10, "m")
	if err != nil {
		t.Fatalf("LexicalChunks: %v", err)
	}
	if len(hits) != 1 || hits[0].DocumentID != "current" {
		t.Fatalf("hits = %+v; want only the current-model document", hits)
	}
}

// TestLexicalChunks_EscapesWildcards proves a term containing LIKE wildcards
// matches literally instead of matching everything.
func TestLexicalChunks_EscapesWildcards(t *testing.T) {
	db := newTestDB(t)
	insertLexDoc(t, db, "literal", "/docs/pct.md", "m",
		database.ChunkRow{Content: "discount 100% off"})
	insertLexDoc(t, db, "other", "/docs/other.md", "m",
		database.ChunkRow{Content: "100x multiplier"})

	hits, err := db.LexicalChunks(nil, []string{"100%"}, 10, "m")
	if err != nil {
		t.Fatalf("LexicalChunks: %v", err)
	}
	if len(hits) != 1 || hits[0].DocumentID != "literal" {
		t.Fatalf("hits = %+v; want only the literal 100%% match", hits)
	}
}

// TestLexicalChunks_NoTerms: an empty or whitespace-only term list must not
// fall through to a full-table match.
func TestLexicalChunks_NoTerms(t *testing.T) {
	db := newTestDB(t)
	insertLexDoc(t, db, "d", "/docs/a.md", "m", database.ChunkRow{Content: "alpha"})

	for _, terms := range [][]string{nil, {}, {"", "  "}} {
		hits, err := db.LexicalChunks(nil, terms, 10, "m")
		if err != nil {
			t.Fatalf("LexicalChunks(%v): %v", terms, err)
		}
		if len(hits) != 0 {
			t.Errorf("LexicalChunks(%v) = %d hits; want 0", terms, len(hits))
		}
	}
}

// TestLexicalChunks_RespectsLimit keeps the candidate pool bounded.
func TestLexicalChunks_RespectsLimit(t *testing.T) {
	db := newTestDB(t)
	for i := 0; i < 5; i++ {
		insertLexDoc(t, db, string(rune('a'+i)), "/guides/jailbreak-"+string(rune('a'+i))+".md", "m",
			database.ChunkRow{Content: "alpha"})
	}
	hits, err := db.LexicalChunks(nil, []string{"jailbreak"}, 2, "m")
	if err != nil {
		t.Fatalf("LexicalChunks: %v", err)
	}
	if len(hits) != 2 {
		t.Errorf("hits = %d; want 2", len(hits))
	}
}
