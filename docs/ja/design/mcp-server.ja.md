# 設計: MCP サーバー (`lite-rag mcp`)

## 1. 概要

lite-rag の RAG パイプラインを **MCP (Model Context Protocol) サーバー** として公開し、
エージェントクライアント（Hermes Agent など）がインデックス済み Markdown ドキュメントを
ツールとして照会できるようにする。

サーバーはサブコマンド（`lite-rag mcp`）として **stdio トランスポート**（JSON-RPC 2.0）
で動作し、MCP クライアントが子プロセスとして起動する。既存の RAG パイプライン
（config → DuckDB → retriever → LLM）をロジック重複なしで再利用する。

---

## 2. スコープ

**対象**
- 新規 cobra サブコマンド `lite-rag mcp`（stdio トランスポート）
- 4 つのツール: `search_docs`、`rag_context`、`list_documents`、`index_documents`
- `cmd/lite-rag/ask.go` から共通 ask/search パイプラインを `internal/rag` へ抽出
- Hermes クライアント登録手順のドキュメント化

**対象外 (YAGNI)**
- HTTP/SSE MCP トランスポート（ローカル起動ツールの標準は stdio）
- ツール結果のストリーミング — MCP `tools/call` は単一のリクエスト/レスポンスであり、
  結果は完成した状態で返される（§8 参照）
- `index` / `reindex` ツール（インデックス作成は CLI の責務のまま）
- 認証（ローカル stdio、クライアントが起動）

---

## 3. アーキテクチャ

```
 MCP クライアント (Hermes Agent)                lite-rag mcp  (子プロセス, stdio)
        │                                            │
        │  initialize / tools/list / tools/call       │
        │ ──────────────────────────────────────────▶ │
        │                                            │  internal/mcpserver/
        │                                            │    (go-sdk: mcp.Server)
        │                                            │      ├─ search_docs
        │                                            │      ├─ rag_context
        │                                            │      ├─ list_documents
        │                                            │      └─ index_documents
        │                                            │            │
        │                                            │    internal/rag/  (Engine)
        │                                            │      ├─ database/   (DuckDB)
        │                                            │      ├─ retriever/  (ベクトル検索)
        │                                            │      └─ llm/        (OpenAI 互換,
        │                                            │                       上流 SSE — 内部)
        │ ◀──── CallToolResult (単一・完成) ───────── │
```

分離: CLI、MCP、および将来の利用者は 1 つのパイプライン実装を共有する。
ツールハンドラは DuckDB や HTTP を一切知らない — `*rag.Engine` のみを見る。

---

## 4. コンポーネント

### 4.1 `cmd/lite-rag/mcp.go` — サブコマンド

- `lite-rag mcp`（継承フラグ `--config`、`--db` を使用）
- 流れ: config 読み込み → `database.Open` → `*rag.Engine` 構築 → `*mcp.Server` 構築 →
  クライアント切断まで `server.Run(ctx, &mcp.StdioTransport{})`
- **stdout はプロトコルチャネル**: 診断出力は全て stderr へ
  （`slog` のデフォルト。`Engine.Progress` は `io.Discard`）

### 4.2 `internal/rag` — 共通パイプライン（`ask.go` から抽出）

```go
// 具象型の狭いサブセット — Engine は実 DuckDB 接続や LLM クライアント
// なしでテスト可能。
type DocDB interface {
    SimilarChunks(query []float32, topK int, embeddingModel string) ([]database.ScoredChunk, error)
    AdjacentChunks(documentID string, lo, hi int) ([]database.ChunkRow, error)
    ListDocuments() ([]database.DocumentRecord, error)
}

type LLM interface {
    Embed(ctx context.Context, texts []string) ([][]float32, error)
    Chat(ctx context.Context, messages []llm.Message, w io.Writer) error
    RewriteQuery(ctx context.Context, query string) ([]string, error)
}

type Engine struct {
    db             DocDB
    llm            LLM
    rewriter       retriever.QueryRewriter // nil = クエリ書き換え無効
    embeddingModel string
    retrievalCfg   config.RetrievalConfig
    Progress       io.Writer              // nil 安全。MCP モードでは io.Discard
}

func NewEngine(db DocDB, client LLM, cfg *config.Config) *Engine

// Search: 検索のみ。topK <= 0 なら設定値を使用。
func (e *Engine) Search(ctx context.Context, query string, topK int) ([]retriever.Passage, error)

// Ask: フルパイプライン。トークンは到着時に w へ流れ、完成した回答も返す
// （CLI は os.Stdout、MCP は io.Discard を渡す）。
func (e *Engine) Ask(ctx context.Context, query string, w io.Writer) (string, []retriever.Passage, error)

// ListDocuments: db.ListDocuments() の薄いラッパー。
func (e *Engine) ListDocuments() ([]database.DocumentRecord, error)
```

nonce タグ付きコンテキストブロックとメッセージ構築は `runAsk` から `Engine.Ask` へ移動する。
Engine はリクエストごとの可変状態を持たない — `retriever` は呼出ごとに構築されるため、
あるリクエストの `top_k` オーバーライドが次へ漏れることはない。
`cmd/lite-rag/ask.go` は `Engine.Ask` を呼ぶようリファクタされる —
**CLI の挙動は変えてはいけない**（テキストモードでは stdout へストリーミング、
`--json` モードではバッファ、ソース出力も同一）。

### 4.3 `internal/mcpserver` — ツール登録

- `New(engine *rag.Engine, ingester Ingester) *mcp.Server` — SDK の総称関数 `mcp.AddTool` で
  型付きハンドラとして 4 ツールを登録
- ツールの `Description` に §5 の利用ガイダンスを記載（エージェント向け
  「いつ使うか」の契約）
- 入出力 struct は `json` + `jsonschema` タグを使用。スキーマは SDK が推論

---

## 5. ツール仕様

| ツール | 入力 | 出力 | LLM 呼出 | 用途 / トリガー（エージェント向け） |
|---|---|---|---|---|
| `search_docs` | `query string`、`top_k int?` | `{passages:[{content, file_path, heading_path, score}]}` | なし | インデックス済み Markdown の意味検索。ファイルパス・見出しパス・スコア付きの生パッセージを返す。情報の所在確認や、自分で読んで合成する場合に使用。 |
| `rag_context` | `question string` | `{answer, sources:[{file_path, heading_path, score}]}` | あり（バッファ） | 質問に対しドキュメント内容のみから生成された完成回答と引用ソースを返す。出来合いの根拠付き回答が欲しい場合に使用。無関係な文脈の場合は「The documents do not contain information to answer this question.」（質問の言語に翻訳）を返す — エラーではない。関連はあるが間接的な文脈は要約し、不足情報を注記する。 |
| `list_documents` | — | `{documents:[{path, chunks, indexed_at, embedding_model}]}` | なし | インデックス済みドキュメントの一覧。検索前にナレッジベースの範囲確認や「何が登録されているか」の回答に使用。 |
| `index_documents` | `path string` | `{path, documents_total, documents_added}` | なし（embed のみ） | ディレクトリ（`*.md` 再帰、変更なしファイルはスキップ）または単一ファイルをナレッジベースに登録。ドキュメント追加・変更後に `search_docs` / `rag_context` で見つかるようにする場合に使用。`documents_added` は新規ファイル数。更新ファイルはその場で更新され 0 と表示される。新規チャンクごとに embed API を呼ぶため、大規模ディレクトリでは時間がかかることがある。 |

---

## 6. データフロー

**`search_docs`** — `query` → `Engine.Search` → embed → `SimilarChunks` →
コンテキスト展開 → パッセージ（スコア降順） → JSON 結果。

**`rag_context`** — `question` → `Engine.Ask` → パッセージ検索 → nonce タグ付き
コンテキストブロック構築 → `client.Chat`（LLM エンドポイントへの SSE は内部・バッファ）
→ 回答 + 重複排除ソース → JSON 結果。

**`list_documents`** — `Engine.ListDocuments` → `db.ListDocuments()` → JSON 結果。

**`index_documents`** — `path` → `Ingester.IndexPath`（`internal/indexer`
のアダプタ、ファイル/ディレクトリを自動判定）→ `db.ListDocuments()` の
前後カウント → JSON 結果。MCP プロセス内で実行され、同時ツール呼出と DuckDB
ハンドルを共有する。このハンドルは `database.idle_timeout_ms` の間アクセスが
ないと解放されるため、サーバーがアイドル中なら CLI の `index` がファイルロックを
取得できる。サーバーは次のリクエストで再オープンし、新しいドキュメントも認識する。
architecture §3.7「プロセス間アクセス」を参照。

---

## 7. エラーハンドリング

- ツール失敗（DB エラー、LLM エラー）→ Go の error として返却 → SDK が
  `IsError` 付き `CallToolResult` に変換
- `search_docs` で 0 件 → **空のパッセージリスト**。エラーではない
- `index_documents` で空パスまたは存在しないパス → Go の error →
  元エラーメッセージを含む `IsError` 結果
- `rag_context` でパッセージなし → 定型文「No relevant documents found in
  the indexed documents.」+ 空の `sources`。**LLM は呼ばれない**
  （engine は nil パッセージを返し `Ask` はチャット呼出をスキップする）。
  「ドキュメントなし」の判定は回答文字列ではなくパッセージリストを基準とする:
  パッセージは見つかったが LLM の完了が空だった場合も、定型文ではなく
  （空の）回答 + ソースを返す
- config/DB の起動失敗 → トランスポート開始前に非ゼロ終了

---

## 8. ストリーミングに関する明確化

MCP `tools/call` は単一のリクエスト/レスポンスであり、部分結果をクライアントへ
ストリーミングするチャネルは存在しない。したがって:

- `rag_context` は完成した回答を**バッファ**し、1 回の結果として返す
- lite-rag と LLM API 間の SSE ストリーミングは変更なし — 内部の上流呼出であり、
  MCP クライアントからは見えない
- `Engine.Ask(ctx, query, w)` は両方の挙動を保持: トークンは到着時に `w` へ流れ
  （CLI ストリーミング）、完成した回答も返る（MCP）

---

## 9. 依存関係

- `github.com/modelcontextprotocol/go-sdk`（Google と共同メンテナンスの公式 Go SDK）
  — サーバー構築、型付きツールハンドラ、stdio トランスポート
- 推移的依存: `github.com/google/jsonschema-go`（スキーマ推論/検証）

いずれも RULES.md  Rule 18 に従い `docs/en/dependencies.md` に記載する。

---

## 10. テスト

**ユニット（`internal/mcpserver`、`internal/rag`）**
- SDK の `InMemoryTransports`: クライアント + サーバーをプロセス内で接続し各ツールを呼出・検証
- フェイク: `DBReader`/`ListDocuments` フェイク、`Embedder` フェイク、`Completer` フェイク
- カバレッジ: 各ツールの正常系 + エラー系（DB エラー、空結果）

**エンドツーエンド（`cmd/lite-rag/mcp_e2e_test.go`）**
- 一時 DuckDB + ローカルの偽 LLM HTTP サーバー（OpenAI 互換 SSE）
- SDK の `CommandTransport` で実バイナリ `lite-rag mcp` を起動
- `initialize` → `tools/list` → `tools/call` のハンドシェイクを 4 ツール全てで検証

**回帰**
- `internal/rag` 抽出後も既存の CLI `ask` テストは green を維持する

---

## 11. Hermes クライアント登録

`~/.hermes/config.yaml` に設定:

```yaml
mcp_servers:
  lite-rag:
    command: /home/isekai/lite-rag/dist/lite-rag
    args: ["mcp", "--config", "<lite-rag config.toml のパス>"]
    enabled: true
    timeout: 120        # LLM の回答には時間がかかる場合がある
```

検証: セッションを再起動し、`search_docs`、`rag_context`、`list_documents` を
インデックス済みデータベースに対して実際に呼び出す。

---

## 12. ドキュメント

- 本設計ドキュメント（`docs/en/design/mcp-server.md`）+ 日本語版
  （`docs/ja/design/mcp-server.ja.md`）（Rule 11）
- MCP の使い方 + Hermes 登録手順を `docs/en/setup.md` と
  `docs/ja/setup.ja.md` に追記（Rule 7）
- 依存関係の記載を `docs/en/dependencies.md` に追記（Rule 18）
