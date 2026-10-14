package main

import (
	"fmt"
	"io"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spf13/cobra"

	"lite-rag/internal/database"
	"lite-rag/internal/llm"
	"lite-rag/internal/mcpserver"
	"lite-rag/internal/rag"
)

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Run as an MCP server (stdio transport) for agent clients",
	Long: "Run lite-rag as a Model Context Protocol server. Communicates over " +
		"stdin/stdout (JSON-RPC 2.0); all diagnostics go to stderr. " +
		"Exposes three tools: search_docs, rag_context, list_documents.",
	RunE: runMCP,
}

func init() {
	rootCmd.AddCommand(mcpCmd)
}

func runMCP(cmd *cobra.Command, args []string) error {
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

	engine := rag.NewEngine(db, client, cfg)
	// stdout is the MCP protocol channel — never write progress there.
	engine.Progress = io.Discard

	server := mcpserver.New(engine)
	return server.Run(cmd.Context(), &mcp.StdioTransport{})
}
