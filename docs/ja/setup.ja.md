# セットアップガイド

## 必要環境

| ツール | バージョン | 用途 |
|---|---|---|
| Go | 1.26.9+ | ビルドツールチェーン |
| make | 任意 | ビルド自動化 |
| cc（clang/gcc） | 任意 | CGo コンパイル（go-duckdb 必須） |
| podman または docker | 任意 | Linux クロスコンパイル（任意） |

> **注意:** ビルドはプロジェクトローカルの Go モジュールキャッシュ（`.go/`）を使用するため、
> `~/go` への書き込みは不要です。Makefile により自動設定されます。

## クイックスタート

```sh
git clone <repo-url> lite-rag
cd lite-rag

# 設定ファイルをコピーして編集
mkdir -p ~/.config/lite-rag
cp config.example.toml ~/.config/lite-rag/config.toml
$EDITOR ~/.config/lite-rag/config.toml

# Git フックとビルドツールをインストール
make setup

# ビルド
make build

# 動作確認
./dist/lite-rag --help
```

## LM Studio の設定

1. LM Studio を起動し、以下のモデルをロードします：
   - 埋め込み: `nomic-ai/nomic-embed-text-v1.5-GGUF`
   - チャット: `openai/gpt-oss-20b`
2. ローカルサーバーを有効化します（デフォルトポート: `1234`）。
3. API が疎通しているか確認します：
   ```sh
   curl http://localhost:1234/v1/models
   ```

## Git フックのインストール

Git フックは `scripts/hooks/` で管理しています。インストール手順：

```sh
make setup
```

`scripts/hooks/pre-commit` と `scripts/hooks/pre-push` を `.git/hooks/` にコピーし、
実行権限を付与します。両フックとも `make check`（lint + test + build）を実行し、
失敗した場合はコミット・プッシュを拒否します。

## リンティング

`make lint` は [golangci-lint](https://golangci-lint.run) v2 を実行します。
ツールは `make setup`（または初回 `make lint` 時）で `.go/bin/` にインストールされます。

lint の挙動は `.golangci.yml` で固定しています。v2 はデフォルトで除外プリセットを
有効にしないため、v1 が自動適用していたプリセット（`comments`、
`std-error-handling`、`common-false-positives`、`legacy`）を明示的に有効にして
います。アップグレード前後で lint ゲートの挙動は変わりません。

## 環境変数

設定ファイルの各項目は環境変数で上書きできます：

| 変数名 | 上書き対象 |
|---|---|
| `LITE_RAG_API_BASE_URL` | `api.base_url` |
| `LITE_RAG_API_KEY` | `api.api_key` |
| `LITE_RAG_API_EMBED_URL` | `api.embed_url` |
| `LITE_RAG_API_EMBED_API_KEY` | `api.embed_api_key` |
| `LITE_RAG_EMBEDDING_MODEL` | `models.embedding` |
| `LITE_RAG_CHAT_MODEL` | `models.chat` |
| `LITE_RAG_DB_PATH` | `database.path` |

### チャットと埋め込みで異なるサーバーを指定する場合

チャット補完と埋め込みを別々のサーバーでホストできます。`api.embed_url` と
`api.embed_api_key` を設定すると、`/embeddings` のみ別サーバーへ送信されます：

```toml
[api]
base_url       = "http://localhost:1234/v1"  # /chat/completions
api_key        = "lm-studio"
embed_url      = "http://localhost:8000/v1"  # /embeddings
embed_api_key  = "embed-key"
```

`embed_url` / `embed_api_key` を空欄にすると `base_url` / `api_key` に
フォールバックするため、既存の設定ファイルは変更なしで動作し続けます。

デフォルトの設定ファイルパスは XDG Base Directory 仕様に従います：
`$XDG_CONFIG_HOME/lite-rag/config.toml`（デフォルト: `~/.config/lite-rag/config.toml`）。
`--config` フラグで上書き可能: `lite-rag --config /path/to/config.toml`。

## クロスコンパイル

### darwin ターゲット（macOS ホストから）

macOS システムの `clang` が `-arch` によるマルチアーキテクチャをサポートしているため、
追加ツールは不要です。

```sh
make cross-build-darwin
# 生成物: dist/lite-rag-darwin-arm64
```

### linux ターゲット（コンテナ経由）

`go-duckdb` は GCC/libstdc++ でビルドされた DuckDB 静的ライブラリをリンクします。
macOS から Linux 向けにクロスコンパイルするには、マッチする C++ ランタイムを持つ
Linux 環境が必要です。コンテナを使った方法が最も簡単です：

```sh
# podman（brew install podman）または docker が必要
make cross-build-linux
# 生成物: dist/lite-rag-linux-amd64  dist/lite-rag-linux-arm64
```

Makefile は `podman` を優先し、なければ `docker` を使用します。

### linux ターゲット（Linux ホスト上で直接）

Linux マシン上で直接ビルドする場合：

```sh
# arm64 クロスコンパイルにはクロスコンパイラが必要
sudo apt-get install gcc-aarch64-linux-gnu

make cross-build-linux-native
```


## プロジェクトローカルモジュールキャッシュ

Makefile が以下を設定します：

```
GOPATH=$(pwd)/.go
GOMODCACHE=$(pwd)/.go/pkg/mod
GOCACHE=$(pwd)/.go/cache
```

ダウンロードしたモジュールとビルド成果物をプロジェクトディレクトリ内に保持します。
ネットワーク制限のある環境で有用です。`.go/` ディレクトリは .gitignore に登録されています。
