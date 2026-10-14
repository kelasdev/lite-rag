// Command mcp runs lite-rag as a Model Context Protocol (MCP) server over
// stdio, so agent clients (e.g. Hermes) can query the indexed documents
// through three tools: search_docs, rag_context, list_documents.
//
// The MCP protocol uses stdin/stdout as a JSON-RPC 2.0 channel, so stdout
// must stay free of any other output — progress and logs go to stderr.
package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spf13/cobra"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
	"lite-rag/internal/indexer"
	"lite-rag/internal/llm"
	"lite-rag/internal/mcpserver"
	"lite-rag/internal/rag"
	"lite-rag/internal/server"
)

var (
	mcpWeb     bool
	mcpWebAddr string
)

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Run as an MCP server (stdio transport) for agent clients",
	Long: "Run lite-rag as a Model Context Protocol server. Communicates over " +
		"stdin/stdout (JSON-RPC 2.0); all diagnostics go to stderr. " +
		"Exposes four tools: search_docs, rag_context, list_documents, " +
		"index_documents. With --web, the Web UI is served from the same process.",
	RunE: runMCP,
}

func init() {
	mcpCmd.Flags().BoolVar(&mcpWeb, "web", false, "also serve the Web UI (HTTP) alongside the MCP server")
	mcpCmd.Flags().StringVar(&mcpWebAddr, "web-addr", "", "Web UI listen address (host:port); overrides config server.addr (requires --web)")
	rootCmd.AddCommand(mcpCmd)
}

func runMCP(cmd *cobra.Command, args []string) error {
	if mcpWebAddr != "" && !mcpWeb {
		return fmt.Errorf("--web-addr requires --web")
	}

	cfg, err := loadConfig()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	db, err := database.Open(cfg.Database.Path)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	client := llm.NewSplit(cfg.API.BaseURL, cfg.API.EffectiveEmbedURL(), cfg.API.APIKey, cfg.API.EffectiveEmbedAPIKey(), cfg.Models.Embedding, cfg.Models.Chat)

	// The Web UI shares this process's single DuckDB connection with the
	// MCP tools, so there is no lock conflict between them.
	if mcpWeb {
		if err := startWeb(cfg, db, client); err != nil {
			return err
		}
	}

	engine := rag.NewEngine(db, client, cfg)
	// stdout is the MCP protocol channel — never write progress there.
	engine.Progress = io.Discard

	// One adapter serves index_documents: dispatch file vs directory,
	// mirroring the index command.
	idx := &pathIndexer{indexer.New(db, client, cfg.Models.Embedding, cfg.Retrieval)}

	server := mcpserver.New(engine, idx)
	return server.Run(cmd.Context(), &mcp.StdioTransport{})
}

// pathIndexer adapts internal/indexer to mcpserver.Ingester: a single
// entry point that stats the path and indexes a directory or one file.
type pathIndexer struct {
	idx *indexer.Indexer
}

func (p *pathIndexer) IndexPath(ctx context.Context, path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("path %s: %w", path, err)
	}
	if info.IsDir() {
		return p.idx.IndexDir(ctx, path)
	}
	return p.idx.IndexFile(ctx, path)
}

// startWeb serves the Web UI in this process and returns once the listener
// accepts connections (or an error if it never comes up, e.g. a bind
// failure). It installs no signal handlers: stdin EOF or SIGINT terminates
// the whole process, which is the correct behavior for a combined
// MCP + Web process.
func startWeb(cfg *config.Config, db *database.DB, client *llm.Client) error {
	addr := cfg.Server.Addr
	if mcpWebAddr != "" {
		addr = mcpWebAddr
	}
	if addr == "" {
		return fmt.Errorf("no Web UI address: set server.addr in the config or pass --web-addr")
	}

	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: parseLogLevel(cfg.Server.LogLevel),
	})))

	srv := server.New(db, client, cfg, version)
	errCh := make(chan error, 1)
	go func() { errCh <- srv.Listen(addr) }()

	// Wait for the listener to accept connections (or fail fast on a
	// bind error) before the MCP transport takes over the process.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case serr := <-errCh:
			return fmt.Errorf("web server: %w", serr)
		default:
		}
		conn, err := net.DialTimeout("tcp", addr, 200*time.Millisecond)
		if err == nil {
			conn.Close()
			// A successful dial only proves *something* listens on the
			// port — when another process holds it, our ListenAndServe
			// fails with a bind error a moment later. Settle briefly so
			// that failure is reported instead of silently serving
			// someone else's listener.
			time.Sleep(200 * time.Millisecond)
			select {
			case serr := <-errCh:
				return fmt.Errorf("web server: %w", serr)
			default:
				return nil
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("web server did not start on %s within 3s", addr)
}
