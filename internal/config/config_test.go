package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"lite-rag/internal/config"
)

func TestDefaults(t *testing.T) {
	// Load from a non-existent path should return defaults without error.
	cfg, err := config.Load("/nonexistent/config.toml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.API.BaseURL != "http://localhost:1234/v1" {
		t.Errorf("BaseURL = %q, want default", cfg.API.BaseURL)
	}
	if cfg.Retrieval.TopK != 5 {
		t.Errorf("TopK = %d, want 5", cfg.Retrieval.TopK)
	}
	if cfg.Retrieval.ContextWindow != 1 {
		t.Errorf("ContextWindow = %d, want 1", cfg.Retrieval.ContextWindow)
	}
}

func TestLoadTOML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")
	content := `
[api]
base_url = "http://example.com/v1"
api_key  = "test-key"

[retrieval]
top_k          = 10
context_window = 2
chunk_size     = 256
chunk_overlap  = 32
min_score      = 0.8
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.API.BaseURL != "http://example.com/v1" {
		t.Errorf("BaseURL = %q", cfg.API.BaseURL)
	}
	if cfg.API.APIKey != "test-key" {
		t.Errorf("APIKey = %q", cfg.API.APIKey)
	}
	if cfg.Retrieval.TopK != 10 {
		t.Errorf("TopK = %d", cfg.Retrieval.TopK)
	}
	if cfg.Retrieval.ContextWindow != 2 {
		t.Errorf("ContextWindow = %d", cfg.Retrieval.ContextWindow)
	}
	if cfg.Retrieval.MinScore != 0.8 {
		t.Errorf("MinScore = %v, want 0.8", cfg.Retrieval.MinScore)
	}
}

func TestMinScoreDefault(t *testing.T) {
	cfg, err := config.Load("/nonexistent/config.toml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Retrieval.MinScore != 0.60 {
		t.Errorf("MinScore = %v, want default 0.60", cfg.Retrieval.MinScore)
	}
}

func TestEnvOverride(t *testing.T) {
	t.Setenv("LITE_RAG_API_BASE_URL", "http://env-override/v1")
	t.Setenv("LITE_RAG_API_KEY", "env-key")

	cfg, err := config.Load("/nonexistent/config.toml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.API.BaseURL != "http://env-override/v1" {
		t.Errorf("BaseURL = %q, want env override", cfg.API.BaseURL)
	}
	if cfg.API.APIKey != "env-key" {
		t.Errorf("APIKey = %q, want env override", cfg.API.APIKey)
	}
}

func TestEnvOverride_DBPath(t *testing.T) {
	t.Setenv("LITE_RAG_DB_PATH", "/tmp/custom.db")

	cfg, err := config.Load("/nonexistent/config.toml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Database.Path != "/tmp/custom.db" {
		t.Errorf("Database.Path = %q, want env override", cfg.Database.Path)
	}
}

func TestInvalidTOML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.toml")
	if err := os.WriteFile(path, []byte("not valid toml :::"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := config.Load(path); err == nil {
		t.Error("Load() expected error for invalid TOML, got nil")
	}
}

func TestEnvOverride_Models(t *testing.T) {
	t.Setenv("LITE_RAG_EMBEDDING_MODEL", "my-embed-model")
	t.Setenv("LITE_RAG_CHAT_MODEL", "my-chat-model")

	cfg, err := config.Load("/nonexistent/config.toml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Models.Embedding != "my-embed-model" {
		t.Errorf("Embedding = %q, want env override", cfg.Models.Embedding)
	}
	if cfg.Models.Chat != "my-chat-model" {
		t.Errorf("Chat = %q, want env override", cfg.Models.Chat)
	}
}

// ── Split chat / embedding endpoints ────────────────────────────────────────

func TestEffectiveEmbed_FallsBackToBaseURL(t *testing.T) {
	// An empty embed_url must fall back to base_url so existing configs keep working.
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")
	content := `
[api]
base_url = "http://chat.example/v1"
api_key  = "chat-key"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := cfg.API.EffectiveEmbedURL(); got != "http://chat.example/v1" {
		t.Errorf("EffectiveEmbedURL() = %q, want base_url fallback", got)
	}
	if got := cfg.API.EffectiveEmbedAPIKey(); got != "chat-key" {
		t.Errorf("EffectiveEmbedAPIKey() = %q, want api_key fallback", got)
	}
}

func TestEffectiveEmbed_UsesSeparateEndpoint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")
	content := `
[api]
base_url       = "http://chat.example/v1"
api_key        = "chat-key"
embed_url      = "http://embed.example/v1"
embed_api_key  = "embed-key"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := cfg.API.EffectiveEmbedURL(); got != "http://embed.example/v1" {
		t.Errorf("EffectiveEmbedURL() = %q, want embed_url", got)
	}
	if got := cfg.API.EffectiveEmbedAPIKey(); got != "embed-key" {
		t.Errorf("EffectiveEmbedAPIKey() = %q, want embed_api_key", got)
	}
	// Chat settings must remain untouched by the embedding overrides.
	if cfg.API.BaseURL != "http://chat.example/v1" {
		t.Errorf("BaseURL = %q, want unchanged", cfg.API.BaseURL)
	}
	if cfg.API.APIKey != "chat-key" {
		t.Errorf("APIKey = %q, want unchanged", cfg.API.APIKey)
	}
}

func TestEnvOverride_EmbedEndpoint(t *testing.T) {
	t.Setenv("LITE_RAG_API_BASE_URL", "http://chat.example/v1")
	t.Setenv("LITE_RAG_API_KEY", "chat-key")
	t.Setenv("LITE_RAG_API_EMBED_URL", "http://embed.example/v1")
	t.Setenv("LITE_RAG_API_EMBED_API_KEY", "embed-key")

	cfg, err := config.Load("/nonexistent/config.toml")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := cfg.API.EffectiveEmbedURL(); got != "http://embed.example/v1" {
		t.Errorf("EffectiveEmbedURL() = %q, want env override", got)
	}
	if got := cfg.API.EffectiveEmbedAPIKey(); got != "embed-key" {
		t.Errorf("EffectiveEmbedAPIKey() = %q, want env override", got)
	}
	if cfg.API.BaseURL != "http://chat.example/v1" {
		t.Errorf("BaseURL = %q, want chat URL unaffected", cfg.API.BaseURL)
	}
}
