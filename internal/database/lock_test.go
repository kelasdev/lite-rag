package database_test

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"lite-rag/internal/database"
)

// DuckDB serialises access per process: a second open in the same process
// joins the existing instance instead of contending for the file lock. The
// conflict this package fixes is therefore only observable across processes,
// so these tests drive a helper process (re-exec of the test binary).

const (
	helperModeEnv = "LITE_RAG_DB_TEST_HELPER"
	helperPathEnv = "LITE_RAG_DB_TEST_PATH"
)

// TestHelperProcess is not a real test: it only runs when the parent test
// spawns this binary with helperModeEnv set.
func TestHelperProcess(t *testing.T) {
	mode := os.Getenv(helperModeEnv)
	if mode == "" {
		return
	}
	path := os.Getenv(helperPathEnv)

	db, err := database.OpenWithOptions(path, database.Options{LockTimeout: 5 * time.Second})
	if err != nil {
		fmt.Fprintln(os.Stderr, "helper open:", err)
		os.Exit(1)
	}
	switch mode {
	case "hold":
		// Signal readiness only after the lock is held, then keep it until
		// the parent closes our stdin.
		fmt.Println("ready")
		_, _ = io.Copy(io.Discard, os.Stdin)
	case "try":
		// Opening is the whole point; the parent inspects the exit code.
	case "insert":
		docID := os.Getenv("LITE_RAG_DB_TEST_DOC")
		doc := database.DocumentRow{
			ID: docID, FilePath: "/docs/" + docID + ".md", FileHash: "hash-" + docID,
			IndexedAt: time.Now().UTC(),
		}
		if err := db.ReplaceDocument(doc, nil); err != nil {
			fmt.Fprintln(os.Stderr, "helper insert:", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintln(os.Stderr, "unknown helper mode", mode)
		os.Exit(2)
	}
	if err := db.Close(); err != nil {
		fmt.Fprintln(os.Stderr, "helper close:", err)
		os.Exit(1)
	}
	os.Exit(0)
}

// ── Helpers ────────────────────────────────────────────────────────────────

func helperCommand(t *testing.T, mode, path string, env ...string) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestHelperProcess$")
	cmd.Env = append(os.Environ(), helperModeEnv+"="+mode, helperPathEnv+"="+path)
	cmd.Env = append(cmd.Env, env...)
	return cmd
}

// startHolder runs a helper process that holds the database lock until stop is
// called.
func startHolder(t *testing.T, path string) (stop func()) {
	t.Helper()
	cmd := helperCommand(t, "hold", path)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("stdin pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start holder: %v", err)
	}
	ready := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(stdout)
		if sc.Scan() {
			ready <- strings.TrimSpace(sc.Text())
			return
		}
		ready <- ""
	}()
	select {
	case msg := <-ready:
		if msg != "ready" {
			_ = stdin.Close()
			_ = cmd.Wait()
			t.Fatalf("holder did not become ready (got %q)", msg)
		}
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("holder did not become ready in 10s")
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			_ = stdin.Close()
			if err := cmd.Wait(); err != nil {
				t.Errorf("holder exit: %v", err)
			}
		})
	}
}

func insertDoc(t *testing.T, db *database.DB, id, path string) {
	t.Helper()
	doc := database.DocumentRow{
		ID: id, FilePath: path, FileHash: "hash-" + id,
		IndexedAt: time.Now().UTC(),
	}
	if err := db.ReplaceDocument(doc, nil); err != nil {
		t.Fatalf("ReplaceDocument(%s): %v", id, err)
	}
}

// ── Lock waiting ───────────────────────────────────────────────────────────

// TestOpen_WaitsForLockRelease proves an open retries instead of failing when
// another process holds the DuckDB file lock for a moment (e.g. an MCP server
// answering a request while a CLI command starts).
func TestOpen_WaitsForLockRelease(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wait.db")
	stop := startHolder(t, path)
	t.Cleanup(stop)

	start := time.Now()
	go func() {
		time.Sleep(300 * time.Millisecond)
		stop()
	}()

	db, err := database.OpenWithOptions(path, database.Options{LockTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("OpenWithOptions while lock released after 300ms: %v", err)
	}
	defer db.Close() //nolint:errcheck

	if elapsed := time.Since(start); elapsed < 250*time.Millisecond {
		t.Errorf("returned after %s; expected to wait for the holder to release", elapsed)
	}
	if err := db.Ping(); err != nil {
		t.Errorf("Ping after acquiring lock: %v", err)
	}
}

// TestOpen_LockTimeoutError proves a permanently held lock surfaces as a
// LockError naming the database, so the user knows why the command failed.
func TestOpen_LockTimeoutError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "timeout.db")
	stop := startHolder(t, path)

	_, err := database.OpenWithOptions(path, database.Options{LockTimeout: 300 * time.Millisecond})
	if err == nil {
		t.Fatal("OpenWithOptions with a held lock: expected an error")
	}
	if !errors.Is(err, database.ErrLocked) {
		t.Errorf("error = %v; want errors.Is(err, ErrLocked)", err)
	}
	msg := err.Error()
	for _, want := range []string{"locked by another process", path} {
		if !strings.Contains(msg, want) {
			t.Errorf("error = %q; want it to contain %q", msg, want)
		}
	}

	// Once the holder exits the same database must open again.
	stop()
	db, err := database.OpenWithOptions(path, database.Options{LockTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("OpenWithOptions after release: %v", err)
	}
	defer db.Close() //nolint:errcheck
}

// ── Idle release ───────────────────────────────────────────────────────────

// TestIdleTimeout_ReleasesLock proves a resident handle stops holding the file
// lock after the idle period, which is what lets the CLI share the database
// with a running MCP server.
func TestIdleTimeout_ReleasesLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idle.db")

	resident, err := database.OpenWithOptions(path, database.Options{IdleTimeout: 100 * time.Millisecond})
	if err != nil {
		t.Fatalf("OpenWithOptions: %v", err)
	}
	defer resident.Close() //nolint:errcheck
	insertDoc(t, resident, "doc1", "/docs/one.md")

	// Poll with a separate process: it can only open once the resident
	// handle has gone idle.
	deadline := time.Now().Add(5 * time.Second)
	for {
		if err := helperCommand(t, "try", path).Run(); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("database still locked after idle timeout")
		}
		time.Sleep(50 * time.Millisecond)
	}

	// The resident handle must keep working after releasing the lock.
	docs, err := resident.ListDocuments()
	if err != nil {
		t.Fatalf("ListDocuments after idle close: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "doc1" {
		t.Errorf("docs = %+v; want doc1", docs)
	}
}

// TestIdleTimeout_ReopensAndSeesNewData proves the idle close is transparent:
// a resident server keeps working and observes rows written by another
// process in the meantime (a never-reopening handle would miss them).
func TestIdleTimeout_ReopensAndSeesNewData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reopen.db")

	resident, err := database.OpenWithOptions(path, database.Options{IdleTimeout: 100 * time.Millisecond})
	if err != nil {
		t.Fatalf("OpenWithOptions: %v", err)
	}
	defer resident.Close() //nolint:errcheck

	insertDoc(t, resident, "doc1", "/docs/one.md")

	// A second process writes after the resident goes idle.
	cmd := helperCommand(t, "insert", path, "LITE_RAG_DB_TEST_DOC=doc2")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("writer process: %v (%s)", err, out)
	}

	docs, err := resident.ListDocuments()
	if err != nil {
		t.Fatalf("ListDocuments after idle reopen: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs = %d; want 2 (resident must see data written while idle)", len(docs))
	}
}

// TestIdleTimeout_InMemoryKeepsData proves the idle close is disabled for an
// in-memory database, whose contents would vanish with the handle.
func TestIdleTimeout_InMemoryKeepsData(t *testing.T) {
	db, err := database.OpenWithOptions("", database.Options{IdleTimeout: 10 * time.Millisecond})
	if err != nil {
		t.Fatalf("OpenWithOptions: %v", err)
	}
	defer db.Close() //nolint:errcheck

	insertDoc(t, db, "doc1", "/docs/one.md")
	time.Sleep(100 * time.Millisecond)

	docs, err := db.ListDocuments()
	if err != nil {
		t.Fatalf("ListDocuments: %v", err)
	}
	if len(docs) != 1 {
		t.Errorf("docs = %d; want 1 (in-memory data must survive)", len(docs))
	}
}

// TestClose_ThenUseFails proves a closed DB reports a clear error instead of
// reopening behind the caller's back.
func TestClose_ThenUseFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "closed.db")

	db, err := database.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := db.ListDocuments(); err == nil {
		t.Fatal("ListDocuments after Close: expected an error")
	}
}
