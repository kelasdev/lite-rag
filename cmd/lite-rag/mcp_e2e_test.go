package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// e2eBinary is the lite-rag binary built once in TestMain.
var e2eBinary string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "lite-rag-mcp-e2e-")
	if err != nil {
		fmt.Fprintln(os.Stderr, "TestMain:", err)
		os.Exit(1)
	}
	e2eBinary = filepath.Join(dir, "lite-rag")
	build := exec.Command("go", "build", "-o", e2eBinary, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		os.RemoveAll(dir)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// TestMCPCmd_E2E indexes a document, then talks MCP to the real binary over
// stdio (spawned via CommandTransport) and exercises all three tools.
func TestMCPCmd_E2E(t *testing.T) {
	srv := mockLLM(t)
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	cfg := writeCfg(t, dir, srv.URL, dbPath)

	docDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(docDir, "doc.md"),
		[]byte("# Topic\n\nSome content about widgets."), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run("--config", cfg, "index", "--dir", docDir); err != nil {
		t.Fatalf("index: %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "e2e-client"}, nil)
	transport := &mcp.CommandTransport{
		Command: exec.Command(e2eBinary, "--config", cfg, "mcp"),
	}
	ctx := context.Background()
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer session.Close()

	// tools/list: exactly the three expected tools.
	var names []string
	for tool, err := range session.Tools(ctx, nil) {
		if err != nil {
			t.Fatalf("list tools: %v", err)
		}
		names = append(names, tool.Name)
	}
	sort.Strings(names)
	want := "index_documents,list_documents,rag_context,search_docs"
	if strings.Join(names, ",") != want {
		t.Fatalf("want tools %s, got %s", want, strings.Join(names, ","))
	}

	// search_docs: the indexed content must come back.
	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "widgets"},
	})
	if err != nil {
		t.Fatalf("search_docs: %v", err)
	}
	if res.IsError {
		t.Fatalf("search_docs error: %s", e2eText(t, res))
	}
	if !strings.Contains(e2eText(t, res), "widgets") {
		t.Errorf("search_docs must return the indexed passage, got: %s", e2eText(t, res))
	}

	// rag_context: mockLLM answers "answer" over SSE.
	res, err = session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "rag_context",
		Arguments: map[string]any{"question": "what is the content about?"},
	})
	if err != nil {
		t.Fatalf("rag_context: %v", err)
	}
	if res.IsError {
		t.Fatalf("rag_context error: %s", e2eText(t, res))
	}
	if !strings.Contains(e2eText(t, res), "answer") {
		t.Errorf("rag_context must return the LLM answer, got: %s", e2eText(t, res))
	}

	// list_documents: doc.md must be listed.
	res, err = session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "list_documents",
		Arguments: map[string]any{},
	})
	if err != nil {
		t.Fatalf("list_documents: %v", err)
	}
	if res.IsError {
		t.Fatalf("list_documents error: %s", e2eText(t, res))
	}
	if !strings.Contains(e2eText(t, res), "doc.md") {
		t.Errorf("list_documents must list doc.md, got: %s", e2eText(t, res))
	}
}

// TestMCPCmd_E2E_Ingest exercises the real indexer adapter through MCP:
// index_documents over stdio must make the document listable and
// searchable in the same process (one DB connection, no lock conflict).
func TestMCPCmd_E2E_Ingest(t *testing.T) {
	srv := mockLLM(t)
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	cfg := writeCfg(t, dir, srv.URL, dbPath)

	docDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(docDir, "ingested.md"),
		[]byte("# Widgets\n\nWidgets are small gadgets.\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "e2e-client"}, nil)
	transport := &mcp.CommandTransport{
		Command: exec.Command(e2eBinary, "--config", cfg, "mcp"),
	}
	ctx := context.Background()
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer session.Close()

	// index_documents: directory dispatch through the real adapter.
	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "index_documents",
		Arguments: map[string]any{"path": docDir},
	})
	if err != nil {
		t.Fatalf("index_documents: %v", err)
	}
	if res.IsError {
		t.Fatalf("index_documents error: %s", e2eText(t, res))
	}
	var idxOut struct {
		DocumentsAdded int `json:"documents_added"`
	}
	if err := json.Unmarshal([]byte(e2eText(t, res)), &idxOut); err != nil {
		t.Fatalf("unmarshal index result: %v (raw: %s)", err, e2eText(t, res))
	}
	if idxOut.DocumentsAdded != 1 {
		t.Errorf("documents_added = %d, want 1", idxOut.DocumentsAdded)
	}

	// The document must now be listed and findable.
	res, err = session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "list_documents",
		Arguments: map[string]any{},
	})
	if err != nil {
		t.Fatalf("list_documents: %v", err)
	}
	if !strings.Contains(e2eText(t, res), "ingested.md") {
		t.Errorf("list_documents must list ingested.md, got: %s", e2eText(t, res))
	}

	res, err = session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "widgets"},
	})
	if err != nil {
		t.Fatalf("search_docs: %v", err)
	}
	if !strings.Contains(e2eText(t, res), "Widgets are small gadgets") {
		t.Errorf("search_docs must find the ingested content, got: %s", e2eText(t, res))
	}

	// A nonexistent path surfaces as an IsError result, not a crash.
	res, err = session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "index_documents",
		Arguments: map[string]any{"path": filepath.Join(docDir, "does-not-exist")},
	})
	if err != nil {
		t.Fatalf("index_documents (missing path): %v", err)
	}
	if !res.IsError {
		t.Errorf("missing path must be an error result, got: %s", e2eText(t, res))
	}
}

// TestMCPCmd_E2E_CLIConcurrency reproduces the cross-process lock conflict:
// an MCP server keeps running while CLI commands open the same database. Both
// must work — the CLI must not fail with "Could not set lock", and the MCP
// server must see documents the CLI indexed after it started.
func TestMCPCmd_E2E_CLIConcurrency(t *testing.T) {
	srv := mockLLM(t)
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	cfg := writeCfg(t, dir, srv.URL, dbPath)

	docDirA := t.TempDir()
	if err := os.WriteFile(filepath.Join(docDirA, "alpha.md"),
		[]byte("# Alpha\n\nAlpha covers widgets.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run("--config", cfg, "index", "--dir", docDirA); err != nil {
		t.Fatalf("index alpha: %v", err)
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "e2e-client"}, nil)
	transport := &mcp.CommandTransport{
		Command: exec.Command(e2eBinary, "--config", cfg, "mcp"),
	}
	ctx := context.Background()
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer session.Close()

	// The MCP server touches the database, so its handle is open right now.
	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "list_documents",
		Arguments: map[string]any{},
	})
	if err != nil {
		t.Fatalf("list_documents: %v", err)
	}
	if !strings.Contains(e2eText(t, res), "alpha.md") {
		t.Fatalf("list_documents must list alpha.md, got: %s", e2eText(t, res))
	}

	// CLI read: must wait for the MCP handle to go idle, not fail on the lock.
	out, err := exec.Command(e2eBinary, "--config", cfg, "docs", "list").CombinedOutput()
	if err != nil {
		t.Fatalf("CLI docs list while MCP is running: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "alpha.md") {
		t.Errorf("CLI docs list must list alpha.md, got: %s", out)
	}

	// CLI write: must also get the lock, index a new document, and release it.
	docDirB := t.TempDir()
	if err := os.WriteFile(filepath.Join(docDirB, "beta.md"),
		[]byte("# Beta\n\nBeta describes gizmo calibration.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err = exec.Command(e2eBinary, "--config", cfg, "index", "--dir", docDirB).CombinedOutput()
	if err != nil {
		t.Fatalf("CLI index while MCP is running: %v\n%s", err, out)
	}

	// The MCP server must see the document indexed after it started: its idle
	// handle reopens instead of serving a stale snapshot.
	res, err = session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "search_docs",
		Arguments: map[string]any{"query": "gizmo calibration"},
	})
	if err != nil {
		t.Fatalf("search_docs: %v", err)
	}
	if res.IsError {
		t.Fatalf("search_docs error: %s", e2eText(t, res))
	}
	if !strings.Contains(e2eText(t, res), "gizmo calibration") {
		t.Errorf("search_docs must find beta.md indexed by the CLI, got: %s", e2eText(t, res))
	}
}

func e2eText(t *testing.T, res *mcp.CallToolResult) string {
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

// TestMCPCmd_E2E_WithWeb runs the MCP server and the Web UI in one process
// (the --web flag): both share the single DuckDB connection, so there is no
// lock conflict between them.
func TestMCPCmd_E2E_WithWeb(t *testing.T) {
	srv := mockLLM(t)
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	cfg := writeCfg(t, dir, srv.URL, dbPath)

	docDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(docDir, "doc.md"),
		[]byte("# Topic\n\nSome content about widgets."), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run("--config", cfg, "index", "--dir", docDir); err != nil {
		t.Fatalf("index: %v", err)
	}

	webAddr := "127.0.0.1:18098"
	client := mcp.NewClient(&mcp.Implementation{Name: "e2e-client"}, nil)
	transport := &mcp.CommandTransport{
		Command: exec.Command(e2eBinary, "--config", cfg, "mcp", "--web", "--web-addr", webAddr),
	}
	ctx := context.Background()
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() {
		session.Close()
		// Wait for the binary to release the web port before the next run.
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			conn, dialErr := net.DialTimeout("tcp", webAddr, 200*time.Millisecond)
			if dialErr != nil {
				return // port free
			}
			conn.Close()
			time.Sleep(100 * time.Millisecond)
		}
		t.Error("web port still held after session close")
	}()

	// MCP side: handshake + tools must work exactly as without --web.
	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      "list_documents",
		Arguments: map[string]any{},
	})
	if err != nil {
		t.Fatalf("list_documents: %v", err)
	}
	if !strings.Contains(e2eText(t, res), "doc.md") {
		t.Errorf("list_documents must list doc.md, got: %s", e2eText(t, res))
	}

	// Web side: the UI must be reachable on the same process.
	httpResp, err := http.Get("http://" + webAddr + "/")
	if err != nil {
		t.Fatalf("web ui not reachable with --web: %v", err)
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode != http.StatusOK {
		t.Errorf("web ui status: want 200, got %d", httpResp.StatusCode)
	}
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		t.Fatalf("read web ui body: %v", err)
	}
	if !strings.Contains(string(body), "lite-rag") {
		t.Errorf("web ui must serve the lite-rag page, got: %.100s", body)
	}
}
