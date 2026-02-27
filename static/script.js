// ================================================================
// TrustMeBro — Multi-Model Scoring + Feedback Frontend
// ================================================================

const API = window.location.origin;

// State
let lastAnalysis = null;
let feedbackLabel = null;

// DOM
const articleText = document.getElementById('article-text');
const urlInput = document.getElementById('url-input');
const fetchBtn = document.getElementById('fetch-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const charCount = document.getElementById('char-count');

// ---- Navigation ----
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const section = link.dataset.section;

        // UI Tabs
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('section-active'));
        document.getElementById(section).classList.add('section-active');

        // Show headlines only on analyzer page
        const headlinesEl = document.getElementById('headlines-section');
        if (headlinesEl) headlinesEl.style.display = section === 'analyzer' ? '' : 'none';

        // Data Refresh
        if (section === 'dashboard') loadDashboard();
        if (section === 'history') loadHistory();
        if (section === 'admin') loadFeedback();
    });
});

articleText.addEventListener('input', () => {
    charCount.textContent = articleText.value.length;
});

// ---- URL Fetching ----
async function fetchUrl() {
    const url = urlInput.value.trim();
    if (!url) { urlInput.focus(); return; }

    fetchBtn.disabled = true;
    fetchBtn.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Fetching...';

    try {
        const res = await fetch(`${API}/fetch-url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Failed (${res.status})`);
        }
        const data = await res.json();
        articleText.value = data.text;
        charCount.textContent = data.text.length;
    } catch (err) {
        showError(err.message);
    } finally {
        fetchBtn.disabled = false;
        fetchBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9"/></svg> Fetch';
    }
}

// ---- Analyze (Multi-Model) ----
let _analyzing = false;
async function analyzeText() {
    if (_analyzing) return; // prevent duplicate requests
    const text = articleText.value.trim();
    if (text.length < 10) { showError('Please enter at least 10 characters.'); return; }

    _analyzing = true;
    hideAll();
    show('result-loading');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;"></div> Analyzing...';

    try {
        const res = await fetch(`${API}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        lastAnalysis = { text, ...data };
        showResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        _analyzing = false;
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> <span>Analyze with All Models</span>';
    }
}

function hideAll() {
    ['result-placeholder', 'result-loading', 'result-error', 'score-cards', 'overall-score'].forEach(id => {
        document.getElementById(id).classList.add('hidden');
    });
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }

function showError(msg) {
    hideAll();
    document.getElementById('error-message').textContent = msg;
    show('result-error');
}

// ---- Render Results ----
function showResults(data) {
    hideAll();
    show('score-cards');
    show('overall-score');

    // Show feedback button
    const feedbackSection = document.getElementById('feedback-section');
    feedbackSection.classList.remove('hidden');
    document.getElementById('feedback-form').classList.add('hidden');
    feedbackLabel = null;
    document.getElementById('btn-fb-true').classList.remove('selected');
    document.getElementById('btn-fb-fake').classList.remove('selected');

    // Render ML score cards
    const modelMap = {
        'logistic_regression': 'lr',
        'svm': 'svm',
        'naive_bayes': 'nb',
        'lightgbm': 'lgbm',
    };

    data.models.forEach(m => {
        const key = modelMap[m.model_name];
        if (!key) return;
        const score = Math.round(m.credibility_score);
        renderScoreCard(key, score, m.label);
    });

    // Render Gemini
    const geminiCardId = 'gemini';
    if (data.gemini.score !== null && data.gemini.score !== undefined) {
        const gs = Math.round(data.gemini.score);
        renderScoreCard(geminiCardId, gs);
        const reasonEl = document.getElementById('gemini-reasoning');
        if (data.gemini.reasoning) {
            reasonEl.textContent = data.gemini.reasoning;
            reasonEl.classList.remove('hidden');
        } else {
            reasonEl.classList.add('hidden');
        }
    } else {
        // Gemini failed
        document.getElementById(`value-${geminiCardId}`).textContent = 'N/A';
        document.getElementById(`ring-${geminiCardId}`).style.strokeDashoffset = '326.7';
        document.getElementById(`badge-${geminiCardId}`).textContent = data.gemini.error ? 'Error' : 'N/A';
        document.getElementById(`badge-${geminiCardId}`).className = 'score-badge uncertain';
        const reasonEl = document.getElementById('gemini-reasoning');
        reasonEl.textContent = data.gemini.error || 'Gemini unavailable';
        reasonEl.classList.remove('hidden');
    }

    // Render ChatGPT
    const chatgptCardId = 'chatgpt';
    if (data.chatgpt.score !== null && data.chatgpt.score !== undefined) {
        const cs = Math.round(data.chatgpt.score);
        renderScoreCard(chatgptCardId, cs);
        const gptReasonEl = document.getElementById('chatgpt-reasoning');
        if (data.chatgpt.reasoning) {
            gptReasonEl.textContent = data.chatgpt.reasoning;
            gptReasonEl.classList.remove('hidden');
        } else {
            gptReasonEl.classList.add('hidden');
        }
    } else {
        document.getElementById(`value-${chatgptCardId}`).textContent = 'N/A';
        document.getElementById(`ring-${chatgptCardId}`).style.strokeDashoffset = '326.7';
        document.getElementById(`badge-${chatgptCardId}`).textContent = data.chatgpt.error ? 'Error' : 'N/A';
        document.getElementById(`badge-${chatgptCardId}`).className = 'score-badge uncertain';
        const gptReasonEl = document.getElementById('chatgpt-reasoning');
        gptReasonEl.textContent = data.chatgpt.error || 'ChatGPT unavailable';
        gptReasonEl.classList.remove('hidden');
    }

    // Render overall score
    const overall = Math.round(data.overall_score);
    const overallColor = getScoreColor(overall);

    document.getElementById('overall-value').textContent = `${overall} / 100`;

    const overallLabel = document.getElementById('overall-label');
    overallLabel.textContent = data.overall_label;
    overallLabel.className = 'overall-label ' + getScoreClass(overall);

    const barFill = document.getElementById('overall-bar-fill');
    barFill.style.background = overallColor;
    setTimeout(() => { barFill.style.width = `${overall}%`; }, 100);
}

function renderScoreCard(key, score, label = null) {
    const ring = document.getElementById(`ring-${key}`);
    const value = document.getElementById(`value-${key}`);
    const badge = document.getElementById(`badge-${key}`);

    const circumference = 326.7; // 2 * pi * 52
    const offset = circumference - (score / 100) * circumference;

    if (label === 'Error') {
        ring.style.stroke = 'var(--text-muted)';
        ring.style.strokeDashoffset = circumference;
        value.textContent = 'N/A';
        badge.textContent = 'ERROR';
        badge.className = 'score-badge low';
        return;
    }

    ring.style.stroke = getScoreColor(score);
    setTimeout(() => { ring.style.strokeDashoffset = offset; }, 50);

    animateValue(value, 0, score, 900, '%');
    badge.textContent = getScoreLabel(score);
    badge.className = 'score-badge ' + getScoreClass(score);
}

function getScoreColor(score) {
    if (score >= 70) return 'var(--green)';
    if (score >= 40) return 'var(--yellow)';
    return 'var(--red)';
}

function getScoreClass(score) {
    if (score >= 70) return 'credible';
    if (score >= 40) return 'uncertain';
    return 'low';
}

function getScoreLabel(score) {
    if (score >= 70) return 'Credible';
    if (score >= 40) return 'Uncertain';
    return 'Low';
}

function animateValue(el, start, end, duration, suffix = '') {
    const range = end - start;
    const startTime = performance.now();
    function update(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = `${Math.round(start + range * eased)}${suffix}`;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ---- Feedback ----
function toggleFeedback() {
    const form = document.getElementById('feedback-form');
    form.classList.toggle('hidden');
}

function setFeedbackLabel(label) {
    feedbackLabel = label;
    document.getElementById('btn-fb-true').classList.toggle('selected', label === 'True');
    document.getElementById('btn-fb-fake').classList.toggle('selected', label === 'Fake');
}

async function submitFeedback() {
    if (!feedbackLabel) { alert('Please select True or Fake verdict.'); return; }
    if (!lastAnalysis) return;

    const btn = document.getElementById('btn-submit-feedback');
    const successMsg = document.getElementById('feedback-success-msg');
    const descArea = document.getElementById('feedback-description');

    btn.disabled = true;
    btn.textContent = 'Submitting...';

    try {
        const payload = {
            article_text: lastAnalysis.text,
            model_scores: lastAnalysis.models,
            gemini_score: lastAnalysis.gemini?.score,
            chatgpt_score: lastAnalysis.chatgpt?.score,
            overall_score: lastAnalysis.overall_score,
            user_label: feedbackLabel,
            user_description: descArea.value.trim(),
        };

        const res = await fetch(`${API}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error('Submission failed');

        // Success UI
        btn.classList.add('hidden');
        document.querySelector('.feedback-label-buttons').classList.add('hidden');
        document.querySelector('.feedback-description-group').classList.add('hidden');
        document.querySelector('.feedback-title').classList.add('hidden');
        successMsg.classList.remove('hidden');

        // Auto-close and reset after delay
        setTimeout(() => {
            toggleFeedback();
            setTimeout(() => {
                btn.classList.remove('hidden');
                btn.disabled = false;
                btn.textContent = 'Submit';
                document.querySelector('.feedback-label-buttons').classList.remove('hidden');
                document.querySelector('.feedback-description-group').classList.remove('hidden');
                document.querySelector('.feedback-title').classList.remove('hidden');
                successMsg.classList.add('hidden');
                descArea.value = '';
                setFeedbackLabel(null);
            }, 400);
        }, 3000);

    } catch (err) {
        alert('Failed: ' + err.message);
        btn.disabled = false;
        btn.textContent = 'Submit';
    }
}

// ---- History ----
async function loadHistory() {
    const listEl = document.getElementById('history-list');
    const countEl = document.getElementById('history-count');
    try {
        const res = await fetch(`${API}/history`);
        const data = await res.json();
        const items = data.history;
        countEl.textContent = `${items.length} ${items.length === 1 ? 'analysis' : 'analyses'}`;
        if (!items.length) {
            listEl.innerHTML = '<div class="history-empty"><p>No analyses yet.</p></div>';
            return;
        }
        listEl.innerHTML = items.map(item => {
            const score = Math.round(item.overall_score || 0);
            const color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--yellow)' : 'var(--red)';
            return `
                <div class="history-item">
                    <span class="history-score-badge" style="background:${color}20;color:${color}">${score}</span>
                    <div class="history-content">
                        <div class="history-text">${escapeHtml(item.text_snippet || '')}</div>
                        <div class="history-meta">
                            <span>${item.overall_label || ''}</span>
                            <span>${formatTimestamp(item.timestamp)}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch { listEl.innerHTML = '<div class="history-empty">Error loading history.</div>'; }
}

async function clearHistory() {
    if (confirm('Clear history?')) { await fetch(`${API}/history`, { method: 'DELETE' }); loadHistory(); }
}

// ---- Admin / Feedback Dashboard ----
async function loadFeedback() {
    const tbody = document.getElementById('feedback-tbody');
    const countEl = document.getElementById('feedback-count');
    try {
        const res = await fetch(`${API}/feedback`);
        const data = await res.json();
        const items = data.feedback;
        countEl.textContent = `${items.length} corrections`;
        if (!items.length) { tbody.innerHTML = '<tr><td colspan="9" class="table-empty">No feedback yet.</td></tr>'; return; }

        tbody.innerHTML = items.map(item => {
            const scores = item.model_scores || [];
            const getS = (name) => {
                const s = scores.find(x => x.model_name === name);
                if (!s) return '—';
                const v = Math.round(s.credibility_score);
                const color = v >= 70 ? 'green' : v >= 40 ? 'yellow' : 'red';
                return `<span class="score-${color}">${v}%</span>`;
            };
            const geminiVal = item.gemini_score != null ? Math.round(item.gemini_score) : '—';
            const geminiHtml = typeof geminiVal === 'number' ? `<span class="score-${geminiVal >= 70 ? 'green' : geminiVal >= 40 ? 'yellow' : 'red'}">${geminiVal}%</span>` : '—';
            const chatgptVal = item.chatgpt_score != null ? Math.round(item.chatgpt_score) : '—';
            const chatgptHtml = typeof chatgptVal === 'number' ? `<span class="score-${chatgptVal >= 70 ? 'green' : chatgptVal >= 40 ? 'yellow' : 'red'}">${chatgptVal}%</span>` : '—';

            return `
                <tr>
                    <td class="admin-article" title="${escapeHtml(item.article_snippet || '')}">${escapeHtml((item.article_snippet || '').slice(0, 40))}…</td>
                    <td>${getS('logistic_regression')}</td>
                    <td>${getS('svm')}</td>
                    <td>${getS('naive_bayes')}</td>
                    <td>${getS('lightgbm')}</td>
                    <td>${geminiHtml}</td>
                    <td>${chatgptHtml}</td>
                    <td><span class="admin-label ${(item.user_label || '').toLowerCase()}">${item.user_label}</span></td>
                    <td class="admin-feedback">${escapeHtml((item.user_description || '').slice(0, 30))}</td>
                    <td>${formatTimestamp(item.timestamp)}</td>
                </tr>
            `;
        }).join('');
    } catch { tbody.innerHTML = '<tr><td colspan="10" class="table-empty">Error loading.</td></tr>'; }
}

async function exportFeedbackCSV() {
    try {
        const res = await fetch(`${API}/feedback`);
        const data = await res.json();
        const items = data.feedback;
        if (!items.length) { alert('No feedback to export.'); return; }

        const headers = ['Timestamp', 'Verdict', 'Explanation', 'LR', 'SVM', 'NB', 'LGBM', 'Gemini', 'ChatGPT', 'Snippet'];
        const csvRows = [headers.join(',')];

        items.forEach(item => {
            const scores = item.model_scores || [];
            const findScore = (name) => {
                const s = scores.find(x => x.model_name === name);
                return s ? Math.round(s.credibility_score) : '';
            };

            const row = [
                `"${item.timestamp}"`,
                `"${item.user_label}"`,
                `"${(item.user_description || '').replace(/"/g, '""')}"`,
                findScore('logistic_regression'),
                findScore('svm'),
                findScore('naive_bayes'),
                findScore('lightgbm'),
                item.gemini_score != null ? Math.round(item.gemini_score) : '',
                item.chatgpt_score != null ? Math.round(item.chatgpt_score) : '',
                `"${(item.article_snippet || '').replace(/"/g, '""')}"`
            ];
            csvRows.push(row.join(','));
        });

        const csvString = csvRows.join('\n');
        const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `trustmebro_feedback_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (err) {
        alert('Export failed: ' + err.message);
    }
}

async function clearFeedback() {
    if (confirm('Clear feedback?')) { await fetch(`${API}/feedback`, { method: 'DELETE' }); loadFeedback(); }
}

// ---- Dashboard Metrics ----
async function loadDashboard() { await loadMetrics(); await loadModels(); }

async function loadMetrics() {
    const tbody = document.getElementById('metrics-tbody');
    if (!tbody) return;
    try {
        const res = await fetch(`${API}/metrics`);
        const data = await res.json();
        const metrics = data.metrics;
        const models = Object.keys(metrics);
        if (!models.length) { tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No metrics.</td></tr>'; return; }

        tbody.innerHTML = models.map(name => {
            const m = metrics[name];
            const roc = m.roc_auc != null ? `${(m.roc_auc * 100).toFixed(1)}%` : 'N/A';
            return `<tr>
                <td style="font-weight:600; color:var(--primary);">${formatModelName(name)}</td>
                <td>${(m.accuracy * 100).toFixed(1)}%</td>
                <td>${(m.precision * 100).toFixed(1)}%</td>
                <td>${(m.recall * 100).toFixed(1)}%</td>
                <td>${(m.f1_score * 100).toFixed(1)}%</td>
                <td>${roc}</td>
            </tr>`;
        }).join('');

        // Stats
        const accs = models.map(m => metrics[m].accuracy);
        const f1s = models.map(m => metrics[m].f1_score);
        document.getElementById('stat-model-count').textContent = models.length;
        document.getElementById('stat-best-acc').textContent = `${(Math.max(...accs) * 100).toFixed(1)}%`;
        document.getElementById('stat-best-f1').textContent = Math.max(...f1s).toFixed(3);
        const best = models.reduce((a, b) => metrics[a].f1_score > metrics[b].f1_score ? a : b);
        document.getElementById('stat-best-model').textContent = formatModelName(best);
    } catch { }
}

async function loadModels() {
    const chips = document.getElementById('model-chips');
    if (!chips) return;
    try {
        const res = await fetch(`${API}/models`);
        const data = await res.json();
        chips.innerHTML = data.available_models.map(n => `<span class="model-chip">${formatModelName(n)}</span>`).join('');
    } catch { }
}

// ---- Headlines ----
let _currentCategory = 'general';

async function loadHeadlines(category = _currentCategory) {
    _currentCategory = category;
    const grid = document.getElementById('headlines-grid');

    grid.innerHTML = '<div class="headlines-loading"><div class="spinner"></div><p>Loading headlines...</p></div>';

    // Update active category button
    document.querySelectorAll('.cat-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.cat === category);
    });

    try {
        const res = await fetch(`${API}/headlines?category=${category}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Error ${res.status}`);
        }
        const data = await res.json();
        renderHeadlines(data.articles);
    } catch (err) {
        grid.innerHTML = `<div class="headlines-error">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <p>${err.message}</p>
        </div>`;
    }
}

function renderHeadlines(articles) {
    const grid = document.getElementById('headlines-grid');
    if (!articles.length) {
        grid.innerHTML = '<div class="headlines-error"><p>No headlines found for this category.</p></div>';
        return;
    }

    grid.innerHTML = articles.map((a, i) => {
        const timeAgo = a.publishedAt ? formatTimestamp(a.publishedAt) : '';
        const imgStyle = a.image ? `background-image: url('${a.image}')` : '';
        const imgClass = a.image ? 'headline-card-img' : 'headline-card-img headline-card-img-empty';
        return `
            <article class="headline-card" onclick="analyzeHeadline('${encodeURIComponent(a.url)}')" style="animation-delay: ${i * 0.05}s">
                <div class="${imgClass}" style="${imgStyle}">
                    ${!a.image ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>' : ''}
                </div>
                <div class="headline-card-body">
                    <div class="headline-source">${escapeHtml(a.source)}</div>
                    <h3 class="headline-card-title">${escapeHtml(a.title)}</h3>
                    ${a.description ? `<p class="headline-card-desc">${escapeHtml(a.description)}</p>` : ''}
                    <div class="headline-card-footer">
                        <span class="headline-time">${timeAgo}</span>
                        <span class="headline-analyze-tag">Click to analyze →</span>
                    </div>
                </div>
            </article>
        `;
    }).join('');
}

async function analyzeHeadline(encodedUrl) {
    const url = decodeURIComponent(encodedUrl);

    // Switch to analyzer view
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelector('.nav-link[data-section="analyzer"]').classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('section-active'));
    document.getElementById('analyzer').classList.add('section-active');

    // Set URL and fetch the article
    urlInput.value = url;
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Auto-fetch the article text
    await fetchUrl();

    // Auto-analyze if text was fetched
    if (articleText.value.trim().length >= 10) {
        analyzeText();
    }
}

// ---- Helpers ----
function formatModelName(n) { return n.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '); }
function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
function formatTimestamp(ts) {
    const d = new Date(ts);
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return d.toLocaleDateString();
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    loadDashboard();
    loadHeadlines();

    // Category tab clicks
    document.querySelectorAll('.cat-btn').forEach(btn => {
        btn.addEventListener('click', () => loadHeadlines(btn.dataset.cat));
    });

    articleText.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeText(); });
    urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') fetchUrl(); });
});
