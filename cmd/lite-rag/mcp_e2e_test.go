package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

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
	want := "list_documents,rag_context,search_docs"
	if strings.Join(names, ",") != want {
		t.Errorf("want tools %s, got %v", want, names)
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
