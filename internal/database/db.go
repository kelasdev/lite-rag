// Package database provides DuckDB connectivity and schema management for lite-rag.
//
// DuckDB permits only one process to hold a database file open for writing;
// every other process — even one that only wants to read — fails with
// "Could not set lock on file". A long-running lite-rag process (mcp, serve)
// would therefore block the CLI for as long as it lives.
//
// DB solves this on both sides:
//
//   - it releases the handle after Options.IdleTimeout of inactivity, so a
//     resident server does not sit on the lock while it is not serving a
//     request, and
//   - every open waits Options.LockTimeout for a competing process to release
//     the lock, retrying with jittered backoff instead of failing immediately.
package database

import (
	"database/sql"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"time"

	_ "github.com/marcboeker/go-duckdb/v2"
)

// DefaultLockTimeout is how long an open waits for another process to release
// the DuckDB file lock when Options.LockTimeout is not set.
const DefaultLockTimeout = 10 * time.Second

// maxLockRetryBackoff caps the sleep between lock retries.
const maxLockRetryBackoff = 500 * time.Millisecond

// ErrLocked reports that the DuckDB file is held by another process and the
// wait timed out. Test with errors.Is.
var ErrLocked = errors.New("duckdb file is locked by another process")

// LockError is returned when an open gives up waiting for the file lock.
type LockError struct {
	Path   string
	Waited time.Duration
	Err    error
}

func (e *LockError) Error() string {
	return fmt.Sprintf(
		"database %s is locked by another process; waited %s for it to be released. "+
			"Stop the other lite-rag process (mcp/serve) or raise database.lock_timeout_ms: %v",
		e.Path, e.Waited.Round(time.Millisecond), e.Err)
}

// Unwrap reports ErrLocked so callers can branch with errors.Is.
func (e *LockError) Unwrap() error { return ErrLocked }

// Options controls how a DB handle is opened and kept.
type Options struct {
	// LockTimeout bounds how long an open (or reopen) waits for another
	// process to release the DuckDB file lock. Zero means DefaultLockTimeout.
	LockTimeout time.Duration

	// IdleTimeout, when greater than zero, closes the underlying handle once
	// the DB has been unused for that long, releasing the file lock so other
	// processes (CLI commands, another MCP server) can open the database.
	// The next database call transparently reopens it. Zero keeps the handle
	// open until Close, which is the right choice for short-lived write
	// commands (index, reindex, docs delete).
	IdleTimeout time.Duration
}

// DB wraps a DuckDB connection whose underlying handle may be released and
// transparently reopened when Options.IdleTimeout elapses.
type DB struct {
	path string
	opts Options

	mu       sync.Mutex
	conn     *sql.DB
	refs     int         // in-flight operations; the handle closes only at 0
	idle     *time.Timer // scheduled idle close, if any
	migrated bool        // schema already ensured by this process
	closed   bool
}

// Open opens (or creates) the DuckDB database at path and applies migrations.
// The handle is kept until Close; use OpenWithOptions for a self-releasing
// handle.
func Open(path string) (*DB, error) {
	return OpenWithOptions(path, Options{})
}

// OpenWithOptions opens (or creates) the DuckDB database at path and applies
// migrations, waiting up to opts.LockTimeout for a competing process.
func OpenWithOptions(path string, opts Options) (*DB, error) {
	if opts.LockTimeout <= 0 {
		opts.LockTimeout = DefaultLockTimeout
	}
	// An in-memory database (empty path) loses its contents when the handle
	// closes, so it must never be released between calls.
	if path == "" {
		opts.IdleTimeout = 0
	}
	db := &DB{path: path, opts: opts}

	db.mu.Lock()
	defer db.mu.Unlock()
	conn, err := db.openConn()
	if err != nil {
		return nil, err
	}
	db.conn = conn
	db.scheduleIdleLocked()
	return db, nil
}

// Close closes the underlying database connection. Later calls fail.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.closed = true
	db.stopIdleLocked()
	if db.conn == nil {
		return nil
	}
	err := db.conn.Close()
	db.conn = nil
	return err
}

// Ping verifies the database connection is alive, reopening it if it was
// released while idle.
func (db *DB) Ping() error {
	conn, err := db.acquire()
	if err != nil {
		return err
	}
	defer db.release()
	return conn.Ping()
}

// QueryRaw executes a raw SQL query and returns the resulting rows.
// Intended for tests and diagnostic use only: the rows are bound to the
// handle current at call time, so do not combine it with Options.IdleTimeout.
func (db *DB) QueryRaw(query string, args ...any) (*sql.Rows, error) {
	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()
	return conn.Query(query, args...)
}

// acquire returns the live handle, opening it if needed, and pins it against
// the idle closer until release is called.
func (db *DB) acquire() (*sql.DB, error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed {
		return nil, fmt.Errorf("database %s: closed", db.path)
	}
	db.stopIdleLocked()
	if db.conn == nil {
		conn, err := db.openConn()
		if err != nil {
			return nil, err
		}
		db.conn = conn
	}
	db.refs++
	return db.conn, nil
}

// release unpins the handle and schedules an idle close when nothing is in
// flight.
func (db *DB) release() {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.refs > 0 {
		db.refs--
	}
	if db.refs == 0 {
		db.scheduleIdleLocked()
	}
}

func (db *DB) stopIdleLocked() {
	if db.idle != nil {
		db.idle.Stop()
		db.idle = nil
	}
}

func (db *DB) scheduleIdleLocked() {
	if db.opts.IdleTimeout <= 0 || db.refs > 0 || db.conn == nil || db.closed {
		return
	}
	db.stopIdleLocked()
	db.idle = time.AfterFunc(db.opts.IdleTimeout, db.closeIfIdle)
}

func (db *DB) closeIfIdle() {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed || db.refs > 0 || db.conn == nil {
		return
	}
	db.idle = nil
	// A close failure leaves the process without a usable handle; the next
	// call reopens from the same path, so the error is not actionable here.
	_ = db.conn.Close()
	db.conn = nil
}

// openConn dials the database, retrying while another process holds the lock.
// The caller must hold db.mu (it reads and writes db.migrated).
func (db *DB) openConn() (*sql.DB, error) {
	start := time.Now()
	deadline := start.Add(db.opts.LockTimeout)
	backoff := 25 * time.Millisecond
	for {
		conn, err := db.dial()
		if err == nil {
			return conn, nil
		}
		if !isLockError(err) {
			return nil, err
		}
		if time.Now().After(deadline) {
			return nil, &LockError{Path: db.path, Waited: time.Since(start), Err: err}
		}
		// Jitter keeps two processes from retrying in lockstep.
		sleep := backoff + time.Duration(rand.Int64N(int64(backoff/2)+1))
		time.Sleep(sleep)
		if backoff < maxLockRetryBackoff {
			backoff *= 2
		}
	}
}

// dial opens and verifies one connection, then applies migrations on the
// first successful open in this process.
func (db *DB) dial() (*sql.DB, error) {
	conn, err := sql.Open("duckdb", db.path)
	if err == nil {
		err = conn.Ping()
	}
	if err == nil && !db.migrated {
		err = migrate(conn)
		if err == nil {
			db.migrated = true
		}
	}
	if err != nil {
		if conn != nil {
			conn.Close()
		}
		return nil, fmt.Errorf("open duckdb %s: %w", db.path, err)
	}
	return conn, nil
}

// isLockError reports whether err is DuckDB refusing the file lock.
func isLockError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "could not set lock") ||
		strings.Contains(msg, "conflicting lock") ||
		strings.Contains(msg, "database is locked")
}

// DocumentRecord holds metadata for a single indexed document.
type DocumentRecord struct {
	ID             string
	FilePath       string
	FileHash       string
	TotalChunks    int
	IndexedAt      string
	EmbeddingModel string
}

// ChunkRecord holds a single chunk with its text content.
type ChunkRecord struct {
	ChunkIndex  int
	HeadingPath string
	Content     string
}

// ListDocuments returns all documents ordered by file_path.
func (db *DB) ListDocuments() ([]DocumentRecord, error) {
	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()

	rows, err := conn.Query(`
		SELECT id, file_path, file_hash, total_chunks,
		       strftime(indexed_at, '%Y-%m-%d %H:%M:%S'), embedding_model
		FROM documents
		ORDER BY file_path
	`)
	if err != nil {
		return nil, fmt.Errorf("list documents: %w", err)
	}
	defer rows.Close()

	var docs []DocumentRecord
	for rows.Next() {
		var d DocumentRecord
		if err := rows.Scan(&d.ID, &d.FilePath, &d.FileHash,
			&d.TotalChunks, &d.IndexedAt, &d.EmbeddingModel); err != nil {
			return nil, fmt.Errorf("scan document: %w", err)
		}
		docs = append(docs, d)
	}
	return docs, rows.Err()
}

// DeleteDocument removes a document and all its chunks. Returns an error if the
// document does not exist.
func (db *DB) DeleteDocument(id string) error {
	conn, err := db.acquire()
	if err != nil {
		return err
	}
	defer db.release()

	// Verify the document exists first.
	var count int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM documents WHERE id = ?`, id).
		Scan(&count); err != nil {
		return fmt.Errorf("check document: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("document %q not found", id)
	}

	if _, err := conn.Exec(`DELETE FROM chunks WHERE document_id = ?`, id); err != nil {
		return fmt.Errorf("delete chunks: %w", err)
	}
	if _, err := conn.Exec(`DELETE FROM documents WHERE id = ?`, id); err != nil {
		return fmt.Errorf("delete document: %w", err)
	}
	return nil
}

// ListStaleDocuments returns documents whose embedding_model differs from
// currentModel, ordered by file_path. Used by the reindex command to find
// documents that need re-embedding after a model change.
func (db *DB) ListStaleDocuments(currentModel string) ([]DocumentRecord, error) {
	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()

	rows, err := conn.Query(`
		SELECT id, file_path, file_hash, total_chunks,
		       strftime(indexed_at, '%Y-%m-%d %H:%M:%S'), embedding_model
		FROM documents
		WHERE embedding_model != ?
		ORDER BY file_path
	`, currentModel)
	if err != nil {
		return nil, fmt.Errorf("list stale documents: %w", err)
	}
	defer rows.Close()

	var docs []DocumentRecord
	for rows.Next() {
		var d DocumentRecord
		if err := rows.Scan(&d.ID, &d.FilePath, &d.FileHash,
			&d.TotalChunks, &d.IndexedAt, &d.EmbeddingModel); err != nil {
			return nil, fmt.Errorf("scan document: %w", err)
		}
		docs = append(docs, d)
	}
	return docs, rows.Err()
}

// UpdateDocumentEmbeddings replaces the embedding vectors for all chunks of a
// document and updates the document's embedding_model, all in one transaction.
// chunks must be ordered by chunk_index and len(chunks) must equal len(embeddings).
func (db *DB) UpdateDocumentEmbeddings(docID, newModel string, chunks []ChunkRecord, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunk count (%d) != embedding count (%d)", len(chunks), len(embeddings))
	}
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

	for i, c := range chunks {
		if _, err := tx.Exec(
			`UPDATE chunks SET embedding = ? WHERE document_id = ? AND chunk_index = ?`,
			embeddings[i], docID, c.ChunkIndex,
		); err != nil {
			return fmt.Errorf("update chunk %d embedding: %w", c.ChunkIndex, err)
		}
	}
	if _, err := tx.Exec(
		`UPDATE documents SET embedding_model = ? WHERE id = ?`, newModel, docID,
	); err != nil {
		return fmt.Errorf("update document model: %w", err)
	}
	return tx.Commit()
}

// DocumentChunks returns all chunks for a document ordered by chunk_index.
func (db *DB) DocumentChunks(id string) ([]ChunkRecord, error) {
	conn, err := db.acquire()
	if err != nil {
		return nil, err
	}
	defer db.release()

	// Verify the document exists first.
	var filePath string
	if err := conn.QueryRow(`SELECT file_path FROM documents WHERE id = ?`, id).
		Scan(&filePath); err == sql.ErrNoRows {
		return nil, fmt.Errorf("document %q not found", id)
	} else if err != nil {
		return nil, fmt.Errorf("check document: %w", err)
	}

	rows, err := conn.Query(`
		SELECT chunk_index, COALESCE(heading_path, ''), content
		FROM chunks
		WHERE document_id = ?
		ORDER BY chunk_index
	`, id)
	if err != nil {
		return nil, fmt.Errorf("query chunks: %w", err)
	}
	defer rows.Close()

	var chunks []ChunkRecord
	for rows.Next() {
		var c ChunkRecord
		if err := rows.Scan(&c.ChunkIndex, &c.HeadingPath, &c.Content); err != nil {
			return nil, fmt.Errorf("scan chunk: %w", err)
		}
		chunks = append(chunks, c)
	}
	return chunks, rows.Err()
}

// migrate creates all required tables if they do not already exist and applies
// incremental schema changes for pre-existing databases.
func migrate(conn *sql.DB) error {
	const ddl = `
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT      PRIMARY KEY,
    file_path       TEXT      NOT NULL,
    file_hash       TEXT      NOT NULL,
    total_chunks    INTEGER   NOT NULL,
    indexed_at      TIMESTAMP NOT NULL,
    embedding_model TEXT      NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT     PRIMARY KEY,
    document_id  TEXT     NOT NULL,  -- logical FK to documents(id); enforced by app
    chunk_index  INTEGER  NOT NULL,
    heading_path TEXT,
    content      TEXT     NOT NULL,
    embedding    FLOAT[]            -- []float32 stored as DuckDB list; requires go-duckdb v2
);
`
	if _, err := conn.Exec(ddl); err != nil {
		return fmt.Errorf("migrate: %w", err)
	}
	// Add embedding_model to pre-existing databases that lack the column.
	const alterDDL = `ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT '';`
	if _, err := conn.Exec(alterDDL); err != nil {
		return fmt.Errorf("migrate alter: %w", err)
	}
	return nil
}
