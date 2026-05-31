/* ==========================================================================
   Veritas AI Premium Controller Script (Vanilla JS)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const textarea = document.getElementById('article-input');
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');
    const validationStatus = document.getElementById('validation-status');
    const btnVerify = document.getElementById('btn-verify');
    const btnClear = document.getElementById('btn-clear');
    const btnPaste = document.getElementById('btn-paste');
    
    // Dashboard States
    const dashboardIdle = document.getElementById('dashboard-idle');
    const dashboardActive = document.getElementById('dashboard-active');
    
    // Forensic Results DOM elements
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugePercentage = document.getElementById('gauge-percentage');
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictDesc = document.getElementById('verdict-desc');
    const consensusVal = document.getElementById('consensus-val');
    
    // Classifiers Elements mapping
    const modelsUI = {
        'Logistic Regression': {
            badge: document.getElementById('badge-lr'),
            fill: document.getElementById('fill-lr'),
            conf: document.getElementById('conf-lr')
        },
        'Decision Tree': {
            badge: document.getElementById('badge-dt'),
            fill: document.getElementById('fill-dt'),
            conf: document.getElementById('conf-dt')
        },
        'Gradient Boosting': {
            badge: document.getElementById('badge-gb'),
            fill: document.getElementById('fill-gb'),
            conf: document.getElementById('conf-gb')
        },
        'Random Forest': {
            badge: document.getElementById('badge-rf'),
            fill: document.getElementById('fill-rf'),
            conf: document.getElementById('conf-rf')
        }
    };
    
    // Features / Keywords
    const keywordsList = document.getElementById('keywords-list');
    
    // Real-Time Live Feed elements
    const liveCoverageBadge = document.getElementById('live-coverage-badge');
    const liveNewsFeed = document.getElementById('live-news-feed');
    
    // History
    const historyRows = document.getElementById('history-rows');
    const historyEmpty = document.getElementById('history-empty');
    const btnClearHistory = document.getElementById('btn-clear-history');
    
    // Local State Variables
    let sessionHistory = [];
    let pasteToggle = 0; // Alternates between True sample and Fake sample
    
    // High-quality Sample Articles for testing
    const SAMPLES = [
        {
            type: "true",
            text: "WASHINGTON (Reuters) - The federal government announced a series of sweeping economic reforms today aimed at curbing inflation and boosting job growth across tech and manufacturing sectors. Legislative leaders from both chambers expressed strong optimism that the bipartisan framework will yield long-term stability. The package includes key tax credits for clean energy investments and standardizations designed to protect supply chain corridors. Economists indicated that these parameters should help moderate market fluctuations over the fiscal quarter, providing a solid foundation for industrial expansions."
        },
        {
            type: "fake",
            text: "SHOCKING NEWS! Whistleblowers have leaked top secret military documents proving that a massive underground alien base is operating in Antarctica, completely hidden from the public eye. According to inside reports, alien structures dating back twelve thousand years were discovered by radar teams last winter. Secret laser defense systems are reportedly active around the perimeter to keep unauthorized explorers from finding the entrance, and corporate leaders have signed confidentiality pacts to conceal this paradigm-shifting technology."
        }
    ];

    // Initialize ledger history from LocalStorage
    function initHistory() {
        const stored = localStorage.getItem('veritas_history');
        if (stored) {
            try {
                sessionHistory = JSON.parse(stored);
                renderHistory();
            } catch (e) {
                sessionHistory = [];
            }
        }
    }

    // Update character and word counters in real-time
    function updateCounters() {
        const text = textarea.value;
        charCount.textContent = text.length;
        
        const cleanText = text.trim();
        const words = cleanText ? cleanText.split(/\s+/).length : 0;
        wordCount.textContent = words;
        
        if (words === 0) {
            validationStatus.className = 'meta-value status-idle';
            validationStatus.textContent = 'Ready';
        } else if (words < 10) {
            validationStatus.className = 'meta-value status-loading';
            validationStatus.textContent = 'Too Short';
        } else {
            validationStatus.className = 'meta-value status-active';
            validationStatus.textContent = 'Optimized';
        }
    }

    textarea.addEventListener('input', updateCounters);

    // Clear textarea and reset active dashboard view back to idle
    btnClear.addEventListener('click', () => {
        textarea.value = '';
        updateCounters();
        dashboardActive.classList.add('hidden');
        dashboardIdle.classList.remove('hidden');
    });

    // Paste sample claimed texts for fast debugging
    btnPaste.addEventListener('click', () => {
        const selectedSample = SAMPLES[pasteToggle];
        textarea.value = selectedSample.text;
        updateCounters();
        
        // Toggle sample for next click
        pasteToggle = (pasteToggle + 1) % SAMPLES.length;
        
        // Trigger subtle glowing pulse on textarea to highlight action
        textarea.focus();
        textarea.classList.add('pulse-glow');
        setTimeout(() => textarea.classList.remove('pulse-glow'), 1000);
    });

    // Perform analysis verification
    btnVerify.addEventListener('click', async () => {
        const text = textarea.value.trim();
        if (!text) {
            alert("Please paste or enter some news text to analyze.");
            return;
        }

        // Lock form during loading state
        btnVerify.classList.add('loading');
        btnVerify.disabled = true;
        textarea.disabled = true;
        btnClear.disabled = true;
        btnPaste.disabled = true;
        validationStatus.textContent = 'Analyzing...';
        validationStatus.className = 'meta-value status-loading';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Server responded with an error.");
            }

            renderDashboard(result, text);
            saveToHistory(result, text);

        } catch (error) {
            console.error(error);
            alert(`Analysis Failed: ${error.message}`);
        } finally {
            // Unlock form fields
            btnVerify.classList.remove('loading');
            btnVerify.disabled = false;
            textarea.disabled = false;
            btnClear.disabled = false;
            btnPaste.disabled = false;
            updateCounters();
        }
    });

    // Dynamic dashboard renderer
    function renderDashboard(data, originalText) {
        // Swap idle graphic card with active panels
        dashboardIdle.classList.add('hidden');
        dashboardActive.classList.remove('hidden');
        
        const consensus = data.consensus;
        const fakePercentage = consensus.fake_percentage;
        const isFake = consensus.label === "Fake News";

        // 1. Update Consensus Circular Progress SVG Gauge
        const circleLength = 314.16; // 2 * PI * 50 radius
        const offsetValue = circleLength - (circleLength * fakePercentage / 100);
        
        // Apply numeric consensus outputs
        gaugePercentage.textContent = `${Math.round(fakePercentage)}%`;
        gaugeFill.style.strokeDashoffset = offsetValue;
        
        // Reset classes and apply matching color schemes (Crimson / Emerald)
        gaugeFill.className.baseVal = `gauge-value-stroke ${isFake ? 'gauge-theme-fake' : 'gauge-theme-true'}`;
        gaugePercentage.className = `gauge-number ${isFake ? 'status-fake' : 'status-true'}`;

        // 2. Set Consolidated Forensic Verdict Badge
        verdictBadge.textContent = consensus.verdict;
        verdictBadge.className = `verdict-badge-box ${isFake ? 'verdict-badge-fake' : 'verdict-badge-true'}`;
        consensusVal.textContent = `${consensus.confidence}%`;

        // Verdict Text descriptions
        if (isFake) {
            verdictDesc.innerHTML = `This claim is flagged as <strong>UNRELIABLE / FABRICATED</strong>. The lexical styles match highly suspicious structural indicators across multiple trained algorithms.`;
        } else {
            if (consensus.live_coverage_found) {
                verdictDesc.innerHTML = `This claim is verified as <strong>REAL NEWS</strong> because active coverage was found on Google News. <br><br><em>Note: Our offline machine learning models flagged this article's writing style as sensationalist or informal (ML Style Fake Score: ${consensus.original_ml_percentage}%), but the live search successfully confirmed the factual event!</em>`;
            } else {
                verdictDesc.innerHTML = `This claim is classified as <strong>VERIFIED / REAL NEWS</strong>. The linguistic syntax and referencing structure align with authenticated journalism patterns.`;
            }
        }

        // 3. Render Independent Classifiers Panel Grid
        for (const [modelName, info] of Object.entries(data.models)) {
            const target = modelsUI[modelName];
            if (target) {
                // Update badge label
                target.badge.textContent = info.label;
                target.badge.className = `model-badge ${info.is_fake ? 'badge-fake' : 'badge-true'}`;
                
                // Animate progress bar widths and colors
                target.fill.className = `confidence-bar-fill ${info.is_fake ? 'fill-fake' : 'fill-true'}`;
                target.fill.style.width = `${info.confidence}%`;
                
                // Apply confidence outputs
                target.conf.textContent = `${info.confidence}%`;
            }
        }

        // 3.5 Render Real-Time Google News Coverage Feed
        liveNewsFeed.innerHTML = '';
        if (consensus.live_coverage_found && data.live_coverage && data.live_coverage.length > 0) {
            liveCoverageBadge.textContent = "Live Coverage Found";
            liveCoverageBadge.className = "live-status-badge live-badge-found";
            
            data.live_coverage.forEach(article => {
                const card = document.createElement('div');
                card.className = 'live-article-card';
                card.innerHTML = `
                    <a href="${article.link}" target="_blank" rel="noopener noreferrer" class="live-article-title-link">
                        ${escapeHtml(article.title)}
                    </a>
                    <div class="live-article-meta">
                        <span class="live-article-source">${escapeHtml(article.source)}</span>
                        <span class="live-article-date">${escapeHtml(article.pub_date)}</span>
                    </div>
                `;
                liveNewsFeed.appendChild(card);
            });
        } else {
            liveCoverageBadge.textContent = "No Live Matches";
            liveCoverageBadge.className = "live-status-badge live-badge-not-found";
            liveNewsFeed.innerHTML = `
                <div class="live-no-matches-box">
                    No matching live article coverage found on Google News. This indicates the story has no matching coverage on standard news streams.
                </div>
            `;
        }

        // 4. Render TF-IDF Significant Keywords Bubble Tags
        keywordsList.innerHTML = '';
        if (data.important_words && data.important_words.length > 0) {
            data.important_words.forEach(item => {
                const badge = document.createElement('div');
                badge.className = 'tag-badge';
                badge.innerHTML = `
                    <span class="tag-word">${escapeHtml(item.word)}</span>
                    <span class="tag-weight">${item.weight.toFixed(1)}</span>
                `;
                keywordsList.appendChild(badge);
            });
        } else {
            keywordsList.innerHTML = `<span class="no-tags">No significant mathematical feature weights captured for this short query.</span>`;
        }
    }

    // Save predictions locally to ledger log memory
    function saveToHistory(data, text) {
        const item = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            preview: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
            fullText: text,
            verdict: data.consensus.verdict,
            is_fake: data.consensus.fake_percentage >= 50,
            confidence: `${data.consensus.confidence}%`,
            fullResult: data
        };

        // Add to array memory
        sessionHistory.unshift(item);
        
        // Keep max 20 logs in local storage to prevent bloating
        if (sessionHistory.length > 20) {
            sessionHistory.pop();
        }

        localStorage.setItem('veritas_history', JSON.stringify(sessionHistory));
        renderHistory();
    }

    // Render session ledger log tables
    function renderHistory() {
        if (sessionHistory.length === 0) {
            historyEmpty.classList.remove('hidden');
            // Hide all other table rows
            const rows = historyRows.querySelectorAll('tr:not(#history-empty)');
            rows.forEach(r => r.remove());
            return;
        }

        historyEmpty.classList.add('hidden');
        
        // Remove existing custom table rows
        const existing = historyRows.querySelectorAll('tr:not(#history-empty)');
        existing.forEach(r => r.remove());

        // Inject dynamic rows
        sessionHistory.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${item.timestamp}</td>
                <td><div class="ledger-preview">${escapeHtml(item.preview)}</div></td>
                <td>
                    <span class="ledger-badge ${item.is_fake ? 'ledger-badge-fake' : 'ledger-badge-true'}">
                        ${item.verdict}
                    </span>
                </td>
                <td>${item.confidence}</td>
                <td><button class="ledger-btn-view" data-id="${item.id}">Inspect</button></td>
            `;
            historyRows.appendChild(tr);
        });

        // Attach event listeners to inspector action buttons
        const inspectBtns = historyRows.querySelectorAll('.ledger-btn-view');
        inspectBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = parseInt(e.target.getAttribute('data-id'));
                const record = sessionHistory.find(r => r.id === id);
                if (record) {
                    textarea.value = record.fullText;
                    updateCounters();
                    renderDashboard(record.fullResult, record.fullText);
                    // Smooth scroll up to top active dashboard panel
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            });
        });
    }

    // Clear saved log history
    btnClearHistory.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear the session prediction logs?")) {
            sessionHistory = [];
            localStorage.removeItem('veritas_history');
            renderHistory();
        }
    });

    // Helper utility to escape HTML inputs safely
    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    // Run initial history loader
    initHistory();
});
