// Command lite-rag is the entry point for the lite-rag CLI.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
)

var (
	cfgFile    string
	dbOverride string
)

var rootCmd = &cobra.Command{
	Use:   "lite-rag",
	Short: "A CLI-based RAG document search system",
}

func init() {
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", config.DefaultConfigPath(), "config file path")
	rootCmd.PersistentFlags().StringVar(&dbOverride, "db", "", "database file path (overrides config database.path)")
}

// loadConfig loads the config file and applies any CLI-level overrides.
func loadConfig() (*config.Config, error) {
	cfg, err := config.Load(cfgFile)
	if err != nil {
		return nil, err
	}
	if dbOverride != "" {
		cfg.Database.Path = dbOverride
	}
	return cfg, nil
}

// openBusyDB opens the database for a process that keeps running after its
// database work is done (mcp, serve, ask). DuckDB allows one process to hold
// the file lock, so these open with an idle timeout: the handle — and with it
// the lock — is released once the process stops touching the database, which
// lets CLI commands run against the same file. The next database call
// reopens it transparently.
func openBusyDB(cfg *config.Config) (*database.DB, error) {
	return database.OpenWithOptions(cfg.Database.Path, database.Options{
		IdleTimeout: cfg.Database.IdleTimeout(),
		LockTimeout: cfg.Database.LockTimeout(),
	})
}

// openCommandDB opens the database for a one-shot command (index, reindex,
// docs). The handle stays open for the command's lifetime, and opening waits
// database.lock_timeout_ms for a resident process to release the lock instead
// of failing immediately.
func openCommandDB(cfg *config.Config) (*database.DB, error) {
	return database.OpenWithOptions(cfg.Database.Path, database.Options{
		LockTimeout: cfg.Database.LockTimeout(),
	})
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
