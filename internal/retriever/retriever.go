// Package retriever implements the retrieval stage of the RAG pipeline.
// It embeds a query, finds the top-K similar chunks via DuckDB, expands each
// hit with adjacent chunks (context window), and deduplicates overlapping spans.
// Optionally a QueryRewriter rewrites the query in parallel for hybrid search.
package retriever

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"unicode"

	"lite-rag/internal/config"
	"lite-rag/internal/database"
	"lite-rag/internal/normalizer"
)

// Embedder is the subset of llm.Client used by the Retriever.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// QueryRewriter rewrites a query into one or more forms more likely to match
// document text. Returning multiple variants (e.g. Japanese and English) causes
// the Retriever to run a parallel search per variant and merge the results.
// If the implementation returns an error, only the original query is searched.
type QueryRewriter interface {
	RewriteQuery(ctx context.Context, query string) ([]string, error)
}

// DBReader is the subset of database.DB used by the Retriever.
type DBReader interface {
	SimilarChunks(query []float32, topK int, embeddingModel string) ([]database.ScoredChunk, error)
	AdjacentChunks(documentID string, lo, hi int) ([]database.ChunkRow, error)
}

// LexicalSearcher is the optional keyword half of hybrid retrieval.
// database.DB implements it; test doubles may omit it, in which case the
// Retriever falls back to pure vector search.
type LexicalSearcher interface {
	LexicalChunks(query []float32, terms []string, limit int, embeddingModel string) ([]database.ScoredChunk, error)
}

// Passage is a retrieved, context-expanded text passage ready for LLM consumption.
type Passage struct {
	Content     string  // merged content of the expanded window
	Score       float32 // relevance: cosine, boosted by keyword evidence (see fuseHits)
	HeadingPath string  // heading hierarchy of the highest-scoring chunk
	DocumentID  string
	FilePath    string // source file path of the document
}

// Retriever executes the vector search and context expansion steps.
type Retriever struct {
	db             DBReader
	emb            Embedder
	rewriter       QueryRewriter // nil = disabled
	topK           int
	candidateK     int // vector/keyword candidates gathered before fusion
	contextWindow  int
	minScore       float32
	lexicalWeight  float32
	lexicalMin     float32
	hybrid         bool
	embeddingModel string
}

// New creates a Retriever configured from cfg.
// rewriter may be nil to disable query rewriting.
// embeddingModel must match the model used during indexing so that vector
// comparisons are only made between embeddings from the same model.
func New(db DBReader, emb Embedder, rewriter QueryRewriter, embeddingModel string, cfg config.RetrievalConfig) *Retriever {
	candidateK := cfg.CandidateK
	if candidateK <= 0 {
		candidateK = cfg.TopK
	}
	if candidateK < cfg.TopK {
		candidateK = cfg.TopK
	}
	return &Retriever{
		db:             db,
		emb:            emb,
		rewriter:       rewriter,
		topK:           cfg.TopK,
		candidateK:     candidateK,
		contextWindow:  cfg.ContextWindow,
		minScore:       cfg.MinScore,
		lexicalWeight:  cfg.LexicalWeight,
		lexicalMin:     cfg.LexicalMin,
		hybrid:         cfg.Hybrid,
		embeddingModel: embeddingModel,
	}
}

// Retrieve embeds query, searches for the top-K similar chunks, expands each hit
// by ±contextWindow adjacent chunks, deduplicates overlapping spans, and returns
// the resulting passages sorted by score descending.
//
// When a QueryRewriter is configured, two searches run concurrently:
//   - original query embedding → SimilarChunks
//   - rewritten query embedding → SimilarChunks
//
// Results are merged (max score per chunk ID) before context expansion.
func (r *Retriever) Retrieve(ctx context.Context, query string) ([]Passage, error) {
	hits, texts, queryVec, err := r.search(ctx, query)
	if err != nil {
		return nil, err
	}

	var lexical []database.ScoredChunk
	if r.hybrid {
		lexical, err = r.lexicalHits(queryVec, texts)
		if err != nil {
			// Keyword search is an enhancement: a failure (e.g. an old schema)
			// must not take retrieval down with it.
			slog.Warn("keyword search failed, using vector hits only", "error", err)
		}
	}

	hits = fuseHits(hits, lexical, r.lexicalWeight, r.minScore, r.lexicalMin, r.topK*max(1, len(texts)))
	if len(hits) == 0 {
		return nil, nil
	}
	return r.expand(hits)
}

// lexicalHits runs the keyword half of hybrid retrieval for every query text
// (the original query plus any rewritten variants). It returns nil when the
// database does not implement LexicalSearcher or the query has no usable terms.
func (r *Retriever) lexicalHits(queryVec []float32, texts []string) ([]database.ScoredChunk, error) {
	searcher, ok := r.db.(LexicalSearcher)
	if !ok {
		return nil, nil
	}
	terms := lexicalTerms(texts...)
	if len(terms) == 0 {
		return nil, nil
	}
	return searcher.LexicalChunks(queryVec, terms, r.candidateK, r.embeddingModel)
}

// fuseHits merges vector hits with keyword hits, applies the admission gate,
// ranks by a keyword-boosted score, and truncates to limit. limit is top_k per
// searched query text: each rewritten variant contributes its own top_k, as
// before hybrid retrieval existed.
//
// Gate: a chunk is admitted when its cosine clears minScore, or when it came
// from keyword search with a match of at least lexicalMin (an exact file-path
// or heading match carries more evidence than a weak cosine on a large corpus).
//
// Score: cos + weight*lex*(1-cos). The lexical signal fills the remaining
// headroom instead of rescaling the cosine, so a hit with no keyword evidence
// keeps its cosine score and the result stays in [0, 1].
func fuseHits(vec, lexical []database.ScoredChunk, weight, minScore, lexicalMin float32, limit int) []database.ScoredChunk {
	type fusedHit struct {
		chunk database.ScoredChunk
		cos   float32
		lex   float32
	}
	byID := make(map[string]*fusedHit, len(vec)+len(lexical))
	for _, h := range vec {
		byID[h.ID] = &fusedHit{chunk: h, cos: h.Score}
	}
	for _, h := range lexical {
		if e, ok := byID[h.ID]; ok {
			e.lex = h.Score
			continue
		}
		// Keyword hits carry their own cosine so they compete on equal terms
		// with the vector candidates.
		byID[h.ID] = &fusedHit{chunk: h, lex: h.Score, cos: h.VectorScore}
	}

	out := make([]database.ScoredChunk, 0, len(byID))
	for _, e := range byID {
		admitted := e.cos >= minScore || (e.lex > 0 && e.lex >= lexicalMin)
		if !admitted {
			continue
		}
		c := e.chunk
		c.Score = e.cos + weight*e.lex*(1-e.cos)
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ID < out[j].ID // deterministic tie-break
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// lexicalTerms extracts keyword terms from the query texts: lowercased
// alphanumeric words of at least three characters, minus a small stopword list,
// capped at maxLexicalTerms and de-duplicated across texts.
func lexicalTerms(texts ...string) []string {
	seen := make(map[string]bool, 8)
	terms := make([]string, 0, 8)
	for _, text := range texts {
		for _, field := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}) {
			if len([]rune(field)) < 3 || lexicalStopwords[field] || seen[field] {
				continue
			}
			seen[field] = true
			terms = append(terms, field)
			if len(terms) == maxLexicalTerms {
				return terms
			}
		}
	}
	return terms
}

// maxLexicalTerms bounds the generated SQL: one GREATEST arm per term.
const maxLexicalTerms = 8

// lexicalStopwords are intent words and function words that appear in almost
// every query and would match unrelated documents. Deliberately small: a rare
// content word is the signal we want to keep.
var lexicalStopwords = map[string]bool{
	// English
	"the": true, "and": true, "for": true, "with": true, "from": true,
	"that": true, "this": true, "how": true, "what": true, "why": true,
	"when": true, "who": true, "are": true, "was": true, "were": true,
	"you": true, "your": true, "its": true, "into": true, "about": true,
	"can": true, "could": true, "should": true, "would": true, "does": true,
	"method": true, "methods": true, "technique": true, "techniques": true,
	// Indonesian
	"dan": true, "yang": true, "untuk": true, "dengan": true, "dari": true,
	"pada": true, "adalah": true, "itu": true, "ini": true, "atau": true,
	"juga": true, "cara": true, "bagaimana": true, "apa": true, "metode": true,
	"sebuah": true, "dalam": true,
}

// search runs concurrent vector searches for the original query and any rewritten
// variants, then merges the results (max score per chunk ID). It also returns
// every query text that was searched, so keyword retrieval can reuse the same
// terms.
func (r *Retriever) search(ctx context.Context, query string) ([]database.ScoredChunk, []string, []float32, error) {
	type searchResult struct {
		hits  []database.ScoredChunk
		texts []string
		vec   []float32
		err   error
	}

	// Hybrid retrieval widens the vector pool so a keyword hit can promote a
	// chunk the vector ranking placed below top_k. Without hybrid the pool is
	// exactly top_k per search, preserving the previous behaviour.
	vectorK := r.topK
	if r.hybrid && r.candidateK > vectorK {
		vectorK = r.candidateK
	}

	// Goroutine 1: embed the original query and search immediately.
	origCh := make(chan searchResult, 1)
	go func() {
		vecs, err := r.emb.Embed(ctx, []string{"search_query: " + normalizer.StripMarkdown(query)})
		if err != nil {
			origCh <- searchResult{err: fmt.Errorf("embed query: %w", err)}
			return
		}
		hits, err := r.db.SimilarChunks(vecs[0], vectorK, r.embeddingModel)
		if err != nil {
			origCh <- searchResult{err: fmt.Errorf("similar chunks: %w", err)}
			return
		}
		origCh <- searchResult{hits: hits, texts: []string{query}, vec: vecs[0]}
	}()

	// Goroutine 2: rewrite the query into variants (e.g. JA + EN), then search
	// each variant in parallel and send the merged result (optional).
	rewriteCh := make(chan searchResult, 1)
	if r.rewriter != nil {
		go func() {
			variants, err := r.rewriter.RewriteQuery(ctx, query)
			if err != nil {
				slog.Warn("query rewrite failed, skipping", "error", err)
				rewriteCh <- searchResult{}
				return
			}
			for i, v := range variants {
				slog.Debug("query rewritten", "original", query, "variant", i, "rewritten", v)
			}

			// Search all variants in parallel.
			type varResult struct{ hits []database.ScoredChunk }
			varChs := make([]chan varResult, len(variants))
			for i, v := range variants {
				ch := make(chan varResult, 1)
				varChs[i] = ch
				go func(v string, ch chan varResult) {
					vecs, err := r.emb.Embed(ctx, []string{"search_query: " + normalizer.StripMarkdown(v)})
					if err != nil {
						slog.Warn("embed rewritten query failed, skipping", "variant", v, "error", err)
						ch <- varResult{}
						return
					}
					hits, err := r.db.SimilarChunks(vecs[0], vectorK, r.embeddingModel)
					if err != nil {
						slog.Warn("similar chunks (rewritten) failed, skipping", "variant", v, "error", err)
						ch <- varResult{}
						return
					}
					ch <- varResult{hits: hits}
				}(v, ch)
			}

			var merged []database.ScoredChunk
			for _, ch := range varChs {
				merged = mergeHits(merged, (<-ch).hits)
			}
			rewriteCh <- searchResult{hits: merged, texts: variants}
		}()
	} else {
		rewriteCh <- searchResult{} // disabled — send empty immediately
	}

	orig := <-origCh
	if orig.err != nil {
		return nil, nil, nil, orig.err
	}
	rewrite := <-rewriteCh // rewrite errors are non-fatal; already logged above

	texts := append(append([]string{}, orig.texts...), rewrite.texts...)
	return mergeHits(orig.hits, rewrite.hits), texts, orig.vec, nil
}

// mergeHits combines two ScoredChunk slices, deduplicating by chunk ID and
// keeping the higher score for any chunk that appears in both.
// The result is sorted by score descending.
func mergeHits(a, b []database.ScoredChunk) []database.ScoredChunk {
	if len(b) == 0 {
		return a
	}
	seen := make(map[string]database.ScoredChunk, len(a)+len(b))
	for _, h := range a {
		seen[h.ID] = h
	}
	for _, h := range b {
		if prev, ok := seen[h.ID]; !ok || h.Score > prev.Score {
			seen[h.ID] = h
		}
	}
	merged := make([]database.ScoredChunk, 0, len(seen))
	for _, h := range seen {
		merged = append(merged, h)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	return merged
}

// expand groups hits by document, computes ±contextWindow spans, merges
// overlapping spans, fetches adjacent chunks, and builds final Passages.
func (r *Retriever) expand(hits []database.ScoredChunk) ([]Passage, error) {
	// Group hits by documentID; also record the file path for each document.
	type hitInfo struct {
		chunkIndex  int
		score       float32
		headingPath string
	}
	docHits := make(map[string][]hitInfo, len(hits))
	docFilePath := make(map[string]string, len(hits))
	for _, h := range hits {
		docHits[h.DocumentID] = append(docHits[h.DocumentID], hitInfo{
			chunkIndex:  h.ChunkIndex,
			score:       h.Score,
			headingPath: h.HeadingPath,
		})
		docFilePath[h.DocumentID] = h.FilePath
	}

	var passages []Passage

	for docID, docHitList := range docHits {
		// Compute expanded spans.
		type span struct {
			lo, hi      int
			score       float32
			headingPath string
		}
		spans := make([]span, len(docHitList))
		for i, h := range docHitList {
			lo := h.chunkIndex - r.contextWindow
			if lo < 0 {
				lo = 0
			}
			spans[i] = span{
				lo:          lo,
				hi:          h.chunkIndex + r.contextWindow,
				score:       h.score,
				headingPath: h.headingPath,
			}
		}

		// Sort by lo, then merge overlapping / adjacent spans.
		sort.Slice(spans, func(i, j int) bool { return spans[i].lo < spans[j].lo })
		merged := []span{spans[0]}
		for _, s := range spans[1:] {
			last := &merged[len(merged)-1]
			if s.lo <= last.hi+1 {
				if s.hi > last.hi {
					last.hi = s.hi
				}
				if s.score > last.score {
					last.score = s.score
					last.headingPath = s.headingPath
				}
			} else {
				merged = append(merged, s)
			}
		}

		// Fetch chunks for each merged span.
		for _, m := range merged {
			chunks, err := r.db.AdjacentChunks(docID, m.lo, m.hi)
			if err != nil {
				return nil, fmt.Errorf("adjacent chunks for %s [%d,%d]: %w", docID, m.lo, m.hi, err)
			}
			var sb strings.Builder
			for i, c := range chunks {
				if i > 0 {
					sb.WriteByte('\n')
				}
				sb.WriteString(c.Content)
			}
			passages = append(passages, Passage{
				Content:     sb.String(),
				Score:       m.score,
				HeadingPath: m.headingPath,
				DocumentID:  docID,
				FilePath:    docFilePath[docID],
			})
		}
	}

	// Return passages sorted by score descending.
	sort.Slice(passages, func(i, j int) bool { return passages[i].Score > passages[j].Score })
	return passages, nil
}
