/* lite-rag Web UI — chat interface */
(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const chat       = document.getElementById('chat');
  const textarea   = document.getElementById('query');
  const submitBtn  = document.getElementById('submit');
  const emptyState = document.getElementById('empty-state');

  // Prevent raw HTML blocks in LLM output from being injected into the DOM.
  // marked v15 removed the built-in sanitize option; we override the html
  // renderer to escape angle brackets instead of passing them through.
  marked.use({
    renderer: {
      html({ text }) { return escHtml(text); },
    },
  });

  // ── Version fetch ─────────────────────────────────────────────────────────
  fetch('/api/status')
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById('version');
      if (el && d.version) el.textContent = 'v' + d.version;
    })
    .catch(() => {});

  // ── Submit on Enter (Shift+Enter = newline) ───────────────────────────────
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitBtn.click();
    }
  });

  // ── Main ask handler ──────────────────────────────────────────────────────
  submitBtn.addEventListener('click', () => {
    const query = textarea.value.trim();
    if (!query) return;

    emptyState.hidden = true;
    addUserMessage(query);
    const assistant = addAssistantMessage();

    textarea.value = '';
    textarea.focus();
    submitBtn.disabled = true;

    let tokenBuffer = '';
    let sourcesReceived = false;

    const fetchOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    };

    fetch('/api/ask', fetchOptions)
      .then(response => {
        if (!response.ok) {
          return response.text().then(t => { throw new Error(t); });
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        function processSSE(text) {
          buf += text;
          const blocks = buf.split('\n\n');
          buf = blocks.pop(); // last (possibly incomplete) block stays in buffer

          for (const block of blocks) {
            if (!block.trim()) continue;

            let eventType = 'message';
            let dataLines = [];

            for (const line of block.split('\n')) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith('data:')) {
                dataLines.push(line.slice(5).trim());
              }
            }

            const data = dataLines.join('\n');

            if (eventType === 'token') {
              try {
                tokenBuffer += JSON.parse(data);
              } catch (_) {
                tokenBuffer += data; // fallback for non-JSON fragments
              }
              assistant.showAnswer(tokenBuffer);
            } else if (eventType === 'sources') {
              try {
                assistant.renderSources(JSON.parse(data));
                sourcesReceived = true;
              } catch (_) {}
            } else if (eventType === 'error') {
              try {
                const err = JSON.parse(data);
                assistant.showError(err.message || 'Unknown error');
              } catch (_) {
                assistant.showError(data);
              }
            } else if (eventType === 'done') {
              // finalize
            }
          }
        }

        function pump() {
          return reader.read().then(({ done, value }) => {
            if (done) return;
            processSSE(decoder.decode(value, { stream: true }));
            return pump();
          });
        }

        return pump();
      })
      .then(() => {
        assistant.finish();
        if (!sourcesReceived) {
          assistant.sources.innerHTML = '';
        }
        submitBtn.disabled = false;
      })
      .catch(err => {
        assistant.showError(err.message);
        assistant.finish();
        submitBtn.disabled = false;
      });
  });

  // ── Message builders ──────────────────────────────────────────────────────
  function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
  }

  function addUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg msg-user';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    msg.appendChild(bubble);
    chat.appendChild(msg);
    scrollToBottom();
  }

  function addAssistantMessage() {
    const msg = document.createElement('div');
    msg.className = 'msg msg-assistant';

    const head = document.createElement('div');
    head.className = 'msg-head';

    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = 'lite-rag';

    const spinner = document.createElement('span');
    spinner.className = 'spinner';

    const toggle = document.createElement('span');
    toggle.className = 'view-toggle';
    const btnRendered = document.createElement('button');
    btnRendered.className = 'toggle-btn active';
    btnRendered.textContent = 'Rendered';
    const btnRaw = document.createElement('button');
    btnRaw.className = 'toggle-btn';
    btnRaw.textContent = 'Raw';
    toggle.appendChild(btnRendered);
    toggle.appendChild(btnRaw);

    head.appendChild(who);
    head.appendChild(spinner);
    head.appendChild(toggle);

    const body = document.createElement('div');
    body.className = 'markdown-body';

    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    cursor.hidden = true;

    const raw = document.createElement('pre');
    raw.className = 'raw-body';
    raw.hidden = true;

    const sources = document.createElement('div');
    sources.className = 'sources';

    msg.appendChild(head);
    msg.appendChild(body);
    msg.appendChild(raw);
    msg.appendChild(sources);
    chat.appendChild(msg);
    scrollToBottom();

    let currentView = 'rendered';

    function setView(view) {
      currentView = view;
      btnRendered.classList.toggle('active', view === 'rendered');
      btnRaw.classList.toggle('active',      view === 'raw');
      body.hidden = (view !== 'rendered');
      raw.hidden  = (view !== 'raw');
    }

    btnRendered.addEventListener('click', () => setView('rendered'));
    btnRaw.addEventListener('click',      () => setView('raw'));

    return {
      body,
      raw,
      sources,

      showAnswer(text) {
        body.hidden = false;
        raw.hidden = true;
        raw.textContent = text;
        body.innerHTML = marked.parse(text);
        cursor.hidden = false;
        body.appendChild(cursor);
        scrollToBottom();
      },

      showError(text) {
        msg.classList.add('error');
        cursor.hidden = true;
        raw.textContent = text;
        raw.hidden = false;
        body.hidden = true;
        btnRendered.classList.remove('active');
        btnRaw.classList.add('active');
        scrollToBottom();
      },

      renderSources(list) {
        if (!list || list.length === 0) return;
        let html = '<div class="sources-label">Sources</div>';
        list.forEach((s, i) => {
          const scoreClass = s.score >= 0.80 ? 'score-high' : s.score >= 0.60 ? 'score-mid' : 'score-low';
          html += `<div class="source-item">
            <span class="num">${i + 1}.</span>
            <span class="filepath">${escHtml(s.file_path)}</span>
            ${s.heading_path ? `<span class="heading">— ${escHtml(s.heading_path)}</span>` : ''}
            <span class="score ${scoreClass}">${s.score.toFixed(3)}</span>
          </div>`;
        });
        sources.innerHTML = html;
        scrollToBottom();
      },

      finish() {
        spinner.remove();
        cursor.hidden = true;
      },
    };
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function escHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
})();
