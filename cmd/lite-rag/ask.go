package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"lite-rag/internal/llm"
	"lite-rag/internal/rag"
	"lite-rag/internal/retriever"
)

var askJSON bool

var askCmd = &cobra.Command{
	Use:   "ask <question>",
	Short: "Answer a question using indexed documents",
	Args:  cobra.MinimumNArgs(1),
	RunE:  runAsk,
}

func init() {
	askCmd.Flags().BoolVar(&askJSON, "json", false, "output answer and sources as JSON")
	rootCmd.AddCommand(askCmd)
}

func runAsk(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	cfg, err := loadConfig()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	db, err := openBusyDB(cfg)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	client := llm.NewSplit(cfg.API.BaseURL, cfg.API.EffectiveEmbedURL(), cfg.API.APIKey, cfg.API.EffectiveEmbedAPIKey(), cfg.Models.Embedding, cfg.Models.Chat)
	engine := rag.NewEngine(db, client, cfg)

	// Suppress progress messages in JSON mode to keep stdout clean.
	progress := io.Writer(os.Stderr)
	if askJSON {
		progress = io.Discard
	}
	engine.Progress = progress

	// In JSON mode the answer is buffered by the engine; in text mode it
	// streams directly to stdout.
	answerWriter := io.Writer(os.Stdout)
	if askJSON {
		answerWriter = io.Discard
	}

	answer, passages, err := engine.Ask(context.Background(), query, answerWriter)
	if err != nil {
		return err
	}
	if len(passages) == 0 {
		if askJSON {
			return outputAnswerJSON("", nil)
		}
		return nil
	}

	if askJSON {
		return outputAnswerJSON(answer, passages)
	}

	// Text mode: print cited sources after the streamed answer.
	fmt.Fprint(os.Stdout, "\n\n---\n")
	n := 0
	printed := make(map[string]bool, len(passages))
	for _, s := range rag.DedupeSources(passages) {
		if printed[s.FilePath] {
			continue
		}
		printed[s.FilePath] = true
		n++
		if s.HeadingPath != "" {
			fmt.Fprintf(os.Stdout, "**Source %d:** %s — %s (score: %.3f)\n",
				n, s.FilePath, s.HeadingPath, s.Score)
		} else {
			fmt.Fprintf(os.Stdout, "**Source %d:** %s (score: %.3f)\n",
				n, s.FilePath, s.Score)
		}
	}
	return nil
}

// outputAnswerJSON writes the answer and deduplicated sources as a JSON object.
func outputAnswerJSON(answer string, passages []retriever.Passage) error {
	type sourceJSON struct {
		FilePath    string  `json:"file_path"`
		HeadingPath string  `json:"heading_path,omitempty"`
		Score       float32 `json:"score"`
	}
	type outputJSON struct {
		Answer  string       `json:"answer"`
		Sources []sourceJSON `json:"sources"`
	}

	sources := make([]sourceJSON, 0)
	for _, s := range rag.DedupeSources(passages) {
		sources = append(sources, sourceJSON{
			FilePath:    s.FilePath,
			HeadingPath: s.HeadingPath,
			Score:       s.Score,
		})
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(outputJSON{Answer: answer, Sources: sources})
}
