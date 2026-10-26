package database

import (
	"database/sql"
	"fmt"
	"strings"
	"time"
)

// DocumentRow represents a row in the documents table.
type DocumentRow struct {
	ID             string
	FilePath       string
	FileHash       string
	TotalChunks    int
	IndexedAt      time.Time
	EmbeddingModel string // model used to generate the chunk embeddings
}

// ChunkRow represents a row in the chunks table.
type ChunkRow struct {
	ID          string
	DocumentID  string
	ChunkIndex  int
	HeadingPath string
	Content     string
	Embedding   []float32
}

// ScoredChunk is a ChunkRow with an associated similarity score and source path.
type ScoredChunk struct {
	ChunkRow
	Score    float32
	FilePath string // file_path of the parent document

	// VectorScore carries the cosine similarity when a row came from a hybrid
	// (keyword) query, whose Score field holds the keyword match instead.
	// Zero for rows from SimilarChunks, where Score is already the cosine.
	VectorScore float32
}

// FindDocumentByPath returns the ID, file hash, and embedding model for the
// document with the given file path. Returns ("", "", "", nil) if not found.
func (db *DB) FindDocumentByPath(filePath string) (id, hash, model string, err error) {
	conn, err := db.acquire()
	if err != nil {
		return "", "", "", err
	}
	defer db.release()

	row := conn.QueryRow(
		"SELECT id, file_hash, embedding_model FROM documents WHERE file_path = ?", filePath)
	err = row.Scan(&id, &hash, &model)
	if err == sql.ErrNoRows {
		return "", "", "", nil
	}
	return
}

// ReplaceDocument atomically replaces a document and all its chunks.
// If a document with the same file_path already exists it is removed first.
func (db *DB) ReplaceDocument(doc DocumentRow, chunks []ChunkRow) error {
	conn, err := db.acquire()
	if err != nil {
		return err
	}
	defer db.release()

	tx, err := conn.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	// Look up existing document ID by file_path to clean up any previous version.
	var existingID string
	row := tx.QueryRow("SELECT id FROM documents WHERE file_path = ?", doc.FilePath)
	if err := row.Scan(&existingID); err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("lookup existing document: %w", err)
	}
	if existingID != "" {
		// Delete chunks before document (logical FK enforced by app code).
		if _, err := tx.Exec(
			"DELETE FROM chunks WHERE document_id = ?", existingID); err != nil {
			return fmt.Errorf("delete old chunks: %w", err)
		}
		if _, err := tx.Exec(
			"DELETE FROM documents WHERE id = ?", existingID); err != nil {
			return fmt.Errorf("delete old document: %w", err)
		}
	}

	// Insert the new document row.
	if _, err := tx.Exec(
		`INSERT INTO documents (id, file_path, file_hash, total_chunks, indexed_at, embedding_model)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		doc.ID, doc.FilePath, doc.FileHash, doc.TotalChunks, doc.IndexedAt, doc.EmbeddingModel); err != nil {
		return fmt.Errorf("insert document: %w", err)
	}

	// Insert chunk rows with []float32 embeddings passed directly to DuckDB (v2).
	for _, c := range chunks {
		if _, err := tx.Exec(
			`INSERT INTO chunks
			   (id, document_id, chunk_index, heading_path, content, embedding)
			 VALUES (?, ?, ?, ?, ?, ?)`,
			c.ID, c.DocumentID, c.ChunkIndex, c.HeadingPath, c.Content,
			c.Embedding); err != nil {
			return fmt.Errorf("insert chunk %d: %w", c.ChunkIndex, err)
		}
	}

	return tx.Commit()
}

// SimilarChunks returns the top-k chunks ranked by cosine similarity to the
// query vector, filtered to documents indexed with the given embeddingModel.
// Computed inside DuckDB using list_cosine_similarity.
func (db *DB) SimilarChunks(query []float32, topK int, embeddingModel string) ([]ScoredChunk, error) {
	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()

	rows, err := conn.Query(
		`SELECT c.id, c.document_id, c.chunk_index, c.heading_path, c.content,
		        list_cosine_similarity(c.embedding, ?) AS score,
		        d.file_path
		 FROM   chunks c
		 JOIN   documents d ON d.id = c.document_id
		 WHERE  c.embedding IS NOT NULL AND len(c.embedding) > 0
		   AND  d.embedding_model = ?
		 ORDER  BY score DESC
		 LIMIT  ?`,
		query, embeddingModel, topK)
	if err != nil {
		return nil, fmt.Errorf("similar chunks query: %w", err)
	}
	defer rows.Close()

	var result []ScoredChunk
	for rows.Next() {
		var sc ScoredChunk
		if err := rows.Scan(
			&sc.ID, &sc.DocumentID, &sc.ChunkIndex, &sc.HeadingPath,
			&sc.Content, &sc.Score, &sc.FilePath); err != nil {
			return nil, fmt.Errorf("scan scored chunk: %w", err)
		}
		result = append(result, sc)
	}
	return result, rows.Err()
}

// LexicalChunks returns chunks whose file path, heading path, or content
// contains any of terms (case-insensitive substring match), restricted to
// documents indexed with embeddingModel and ordered by match strength.
//
// This is the keyword half of hybrid retrieval. Vector search alone ranks a
// document titled "Jailbreaking-Master-Guide" below generic security notes
// that merely sit nearby in embedding space; an exact term match in the path
// or heading is a strong, cheap signal that the document is about the query.
//
// Score is the keyword match score in [0, 1]: 0.8 for a file-path match, 0.6
// for a heading match, 0.25 for a content match, taking the strongest term.
// VectorScore is the cosine similarity of the same chunk when query is
// non-nil, so the caller can fuse both signals without a second query.
// Empty terms or a non-positive limit yield no hits.
func (db *DB) LexicalChunks(query []float32, terms []string, limit int, embeddingModel string) ([]ScoredChunk, error) {
	terms = cleanTerms(terms)
	if len(terms) == 0 || limit <= 0 {
		return nil, nil
	}

	// Build one GREATEST(...) arm per term: the strongest column match wins,
	// so a long query still gets credit for the term that names the document.
	scoreArms := make([]string, 0, len(terms))
	whereArms := make([]string, 0, len(terms))
	scoreArgs := make([]any, 0, len(terms)*3)
	whereArgs := make([]any, 0, len(terms)*3)
	for _, t := range terms {
		pattern := "%" + escapeLike(t) + "%"
		scoreArms = append(scoreArms, `GREATEST(
			CASE WHEN lower(d.file_path) LIKE ? ESCAPE '\' THEN 0.8 ELSE 0 END,
			CASE WHEN lower(COALESCE(c.heading_path, '')) LIKE ? ESCAPE '\' THEN 0.6 ELSE 0 END,
			CASE WHEN lower(c.content) LIKE ? ESCAPE '\' THEN 0.25 ELSE 0 END)`)
		scoreArgs = append(scoreArgs, pattern, pattern, pattern)
		whereArms = append(whereArms, `(lower(d.file_path) LIKE ? ESCAPE '\'
			OR lower(COALESCE(c.heading_path, '')) LIKE ? ESCAPE '\'
			OR lower(c.content) LIKE ? ESCAPE '\')`)
		whereArgs = append(whereArgs, pattern, pattern, pattern)
	}

	// Carry the cosine alongside the keyword score so fusion does not have to
	// fetch candidates a second time.
	cosExpr := `CAST(0 AS FLOAT)`
	cosArgs := []any{}
	if len(query) > 0 {
		cosExpr = `COALESCE(list_cosine_similarity(c.embedding, ?), 0.0)`
		cosArgs = append(cosArgs, query)
	}

	querySQL := `SELECT c.id, c.document_id, c.chunk_index, COALESCE(c.heading_path, ''),
	                    c.content, GREATEST(` + strings.Join(scoreArms, ", ") + `) AS lex,
	                    ` + cosExpr + ` AS cos, d.file_path
	             FROM chunks c
	             JOIN documents d ON d.id = c.document_id
	             WHERE d.embedding_model = ?
	               AND (` + strings.Join(whereArms, " OR ") + `)
	             ORDER BY lex DESC, cos DESC, c.chunk_index ASC, c.id ASC
	             LIMIT ?`

	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()

	args := make([]any, 0, len(scoreArgs)+len(cosArgs)+len(whereArgs)+2)
	args = append(args, scoreArgs...)
	args = append(args, cosArgs...)
	args = append(args, embeddingModel)
	args = append(args, whereArgs...)
	args = append(args, limit)

	rows, err := conn.Query(querySQL, args...)
	if err != nil {
		return nil, fmt.Errorf("lexical chunks query: %w", err)
	}
	defer rows.Close()

	var result []ScoredChunk
	for rows.Next() {
		var sc ScoredChunk
		if err := rows.Scan(
			&sc.ID, &sc.DocumentID, &sc.ChunkIndex, &sc.HeadingPath,
			&sc.Content, &sc.Score, &sc.VectorScore, &sc.FilePath); err != nil {
			return nil, fmt.Errorf("scan lexical chunk: %w", err)
		}
		result = append(result, sc)
	}
	return result, rows.Err()
}

// cleanTerms lowercases, trims, and de-duplicates search terms, dropping
// empties while preserving order.
func cleanTerms(terms []string) []string {
	seen := make(map[string]struct{}, len(terms))
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		t = strings.ToLower(strings.TrimSpace(t))
		if t == "" {
			continue
		}
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		out = append(out, t)
	}
	return out
}

// escapeLike escapes LIKE wildcards so a term containing % or _ matches
// literally. Used together with ESCAPE '\' in the query.
func escapeLike(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch r {
		case '\\', '%', '_':
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

// AdjacentChunks returns the chunks of documentID whose chunk_index is in
// [lo, hi], ordered by chunk_index. Used for context window expansion.
func (db *DB) AdjacentChunks(documentID string, lo, hi int) ([]ChunkRow, error) {
	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()

	rows, err := conn.Query(
		`SELECT id, document_id, chunk_index, heading_path, content
		 FROM chunks
		 WHERE document_id = ?
		   AND chunk_index BETWEEN ? AND ?
		 ORDER BY chunk_index ASC`,
		documentID, lo, hi)
	if err != nil {
		return nil, fmt.Errorf("query adjacent chunks: %w", err)
	}
	defer rows.Close()

	var result []ChunkRow
	for rows.Next() {
		var c ChunkRow
		if err := rows.Scan(
			&c.ID, &c.DocumentID, &c.ChunkIndex, &c.HeadingPath, &c.Content); err != nil {
			return nil, fmt.Errorf("scan adjacent chunk: %w", err)
		}
		result = append(result, c)
	}
	return result, rows.Err()
}
