document.addEventListener("DOMContentLoaded", () => {
    // ---- Hero fade + top bar reveal on scroll ----
    const hero = document.getElementById("hero");
    const topBar = document.getElementById("top-bar");

    if (hero && topBar) {
        let heroH = hero.offsetHeight;
        let ticking = false;

        window.addEventListener("resize", () => { heroH = hero.offsetHeight; }, { passive: true });

        window.addEventListener("scroll", () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    const progress = Math.min(window.scrollY / (heroH * 0.6), 1);
                    hero.style.opacity = 1 - progress;
                    topBar.classList.toggle("visible", progress > 0.15);
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });
    }

    const f1Input = document.getElementById("fighter1");
    const f2Input = document.getElementById("fighter2");
    const ac1 = document.getElementById("autocomplete1");
    const ac2 = document.getElementById("autocomplete2");


    let allFighters = [];

    fetch("/api/fighters")
        .then(res => res.json())
        .then(data => { allFighters = data.fighters || []; })
        .catch(err => console.error(err));

    fetch("/api/model-info")
        .then(res => res.json())
        .then(info => {
            const el = (id, val) => {
                const e = document.getElementById(id);
                if (e) e.textContent = val;
            };
            el("fighter-count", info.fighterCount.toLocaleString());
            el("fight-count", info.fightCount.toLocaleString());
            el("feature-count", info.featureCount);
        })
        .catch(err => console.error(err));

    // ---- Autocomplete ----
    function setupAutocomplete(inputEl, listEl) {
        let activeIdx = -1;

        function updateActive() {
            const items = listEl.querySelectorAll("li");
            items.forEach((li, i) => {
                li.classList.toggle("active", i === activeIdx);
                li.setAttribute("aria-selected", i === activeIdx);
            });
        }

        function selectItem(match) {
            inputEl.value = match;
            listEl.innerHTML = "";
            listEl.classList.add("hidden");
            inputEl.setAttribute("aria-expanded", "false");
            activeIdx = -1;
        }

        inputEl.addEventListener("input", function () {
            const val = this.value;
            listEl.innerHTML = "";
            activeIdx = -1;
            if (!val) { listEl.classList.add("hidden"); inputEl.setAttribute("aria-expanded", "false"); return; }
            const matches = allFighters
                .filter(f => f.toLowerCase().includes(val.toLowerCase()))
                .slice(0, 10);
            if (matches.length > 0) {
                listEl.classList.remove("hidden");
                inputEl.setAttribute("aria-expanded", "true");
                const needle = val.toLowerCase();
                matches.forEach(match => {
                    const li = document.createElement("li");
                    li.setAttribute("role", "option");
                    li.setAttribute("aria-selected", "false");
                    const idx = match.toLowerCase().indexOf(needle);
                    if (idx >= 0) {
                        li.appendChild(document.createTextNode(match.slice(0, idx)));
                        const strong = document.createElement("strong");
                        strong.textContent = match.slice(idx, idx + needle.length);
                        li.appendChild(strong);
                        li.appendChild(document.createTextNode(match.slice(idx + needle.length)));
                    } else {
                        li.textContent = match;
                    }
                    li.addEventListener("click", () => selectItem(match));
                    listEl.appendChild(li);
                });
            } else {
                listEl.classList.add("hidden");
                inputEl.setAttribute("aria-expanded", "false");
            }
        });

        inputEl.addEventListener("keydown", e => {
            const items = listEl.querySelectorAll("li");
            if (!items.length) return;
            if (e.key === "ArrowDown") {
                e.preventDefault();
                activeIdx = Math.min(activeIdx + 1, items.length - 1);
                updateActive();
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                activeIdx = Math.max(activeIdx - 1, 0);
                updateActive();
            } else if (e.key === "Enter" && activeIdx >= 0) {
                e.preventDefault();
                selectItem(items[activeIdx].textContent);
            } else if (e.key === "Escape") {
                listEl.classList.add("hidden");
                inputEl.setAttribute("aria-expanded", "false");
                activeIdx = -1;
            }
        });

        document.addEventListener("click", e => {
            if (e.target !== inputEl && !listEl.contains(e.target)) {
                listEl.classList.add("hidden");
                inputEl.setAttribute("aria-expanded", "false");
                activeIdx = -1;
            }
        });
    }

    setupAutocomplete(f1Input, ac1);
    setupAutocomplete(f2Input, ac2);

    const resultSec = document.getElementById("result-section");

    const revealObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) entry.target.classList.add("visible");
        });
    }, { threshold: 0.1 });
    document.querySelectorAll(".reveal").forEach(el => revealObserver.observe(el));

    // ---- Prediction ----
    const matchBtn = document.getElementById("predict-btn");
    const btnText = document.querySelector(".btn-text");
    const loader = document.querySelector(".loader");

    const LOADING_STEPS = [
        "loading fighter profiles",
        "building 86-feature vector",
        "standardizing against training set",
        "xgboost voting",
        "lightgbm voting",
        "logistic regression voting",
        "random forest voting",
        "svm voting",
        "averaging ensemble",
        "platt-scaling probabilities",
        "computing shap attribution",
        "finding nearest neighbors",
        "finalizing odds",
    ];
    const MIN_LOAD_MS = 2000;
    const predictionEl = document.getElementById("prediction-top");
    const loadingStatusEl = document.getElementById("loading-status");

    matchBtn.addEventListener("click", () => {
        const fighter1 = f1Input.value.trim();
        const fighter2 = f2Input.value.trim();
        if (!fighter1 || !fighter2) { alert("Please select both fighters."); return; }
        if (fighter1.toLowerCase() === fighter2.toLowerCase()) { alert("Pick two different fighters."); return; }

        const started = Date.now();

        resultSec.classList.add("hidden");
        btnText.classList.add("hidden");
        loader.classList.remove("hidden");
        matchBtn.disabled = true;
        predictionEl.classList.remove("idle");
        predictionEl.classList.add("loading");

        // Scroll the prediction strip into view so the status line is visible
        predictionEl.scrollIntoView({ behavior: "smooth", block: "start" });

        // Cycle loading status, restart tick animation on each change
        let stepIdx = 0;
        if (loadingStatusEl) loadingStatusEl.textContent = LOADING_STEPS[0];
        const statusTimer = setInterval(() => {
            stepIdx = (stepIdx + 1) % LOADING_STEPS.length;
            if (loadingStatusEl) {
                loadingStatusEl.textContent = LOADING_STEPS[stepIdx];
                loadingStatusEl.style.animation = "none";
                void loadingStatusEl.offsetWidth;
                loadingStatusEl.style.animation = "";
            }
        }, 190);

        const finish = (cb) => {
            const elapsed = Date.now() - started;
            const wait = Math.max(0, MIN_LOAD_MS - elapsed);
            setTimeout(() => {
                clearInterval(statusTimer);
                predictionEl.classList.remove("loading");
                btnText.classList.remove("hidden");
                loader.classList.add("hidden");
                matchBtn.disabled = false;
                cb && cb();
            }, wait);
        };

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fighter1, fighter2 })
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    finish(() => { predictionEl.classList.add("idle"); alert("Error: " + data.error); });
                } else {
                    finish(() => renderResults(data));
                }
            })
            .catch(err => {
                console.error(err);
                finish(() => { predictionEl.classList.add("idle"); alert("Network error."); });
            });
    });

    function esc(str) {
        const d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function fmt(val, suffix) {
        if (val === null || val === undefined) return "\u2014";
        return suffix ? esc(val + suffix) : esc(String(val));
    }

    // Convert win probability to American moneyline odds.
    function toMoneyline(prob) {
        if (prob >= 0.999) return "-∞";
        if (prob <= 0.001) return "+∞";
        if (prob >= 0.5) return "\u2212" + Math.round((100 * prob) / (1 - prob));
        return "+" + Math.round((100 * (1 - prob)) / prob);
    }

    // ---- Render Results ----
    function renderResults(data) {
        const { fighter1, fighter2, winner, f1Prob, f2Prob, f1, f2, analysis } = data;
        const p1 = (f1Prob * 100).toFixed(1);
        const p2 = (f2Prob * 100).toFixed(1);

        // Probability bar
        document.getElementById("prob-f1-name").textContent = fighter1;
        document.getElementById("prob-f2-name").textContent = fighter2;
        document.getElementById("prob-f1-val").textContent = p1 + "%";
        document.getElementById("prob-f2-val").textContent = p2 + "%";

        // Moneyline odds
        const oddsEl1 = document.getElementById("odds-f1");
        const oddsEl2 = document.getElementById("odds-f2");
        if (oddsEl1) oddsEl1.textContent = toMoneyline(f1Prob);
        if (oddsEl2) oddsEl2.textContent = toMoneyline(f2Prob);

        const f1Fill = document.getElementById("prob-f1-fill");
        const f2Fill = document.getElementById("prob-f2-fill");
        f1Fill.style.width = "0%";
        f2Fill.style.width = "0%";

        document.getElementById("winner-name").textContent = winner;
        const arrowL = document.getElementById("winner-arrow");
        const arrowR = document.getElementById("winner-arrow-r");
        arrowL.classList.toggle("active", winner === fighter1);
        arrowR.classList.toggle("active", winner === fighter2);

        // Fighter profiles
        renderProfile("f1", fighter1, f1);
        renderProfile("f2", fighter2, f2);

        // Comparison bars
        renderComparison(fighter1, fighter2, f1, f2);

        // Win methods
        renderMethods(f1, f2);

        // Analysis sections
        if (analysis) {
            renderKeyFactors(analysis.keyFactors, fighter1, fighter2);
            renderCategoryEdges(analysis.categories, fighter1, fighter2);
            renderModelBreakdown(analysis.modelBreakdown, fighter1, fighter2);
            renderHistorical(analysis.historicalMatchups, fighter1, fighter2);
        }

        // Reveal result section and animate bar fill
        resultSec.classList.remove("hidden");
        setTimeout(() => {
            f1Fill.style.width = p1 + "%";
            f2Fill.style.width = p2 + "%";
        }, 80);

        // Scroll so the verdict strip sits at the top of the viewport
        const predEl = document.getElementById("prediction-top");
        if (predEl) predEl.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ---- Fighter Profile ----
    function renderProfile(prefix, name, stats) {
        document.getElementById(`profile-${prefix}-name`).textContent = name;
        document.getElementById(`profile-${prefix}-record`).textContent = stats.record;
        document.getElementById(`profile-${prefix}-stance`).textContent = stats.stance || "Unknown";
        document.getElementById(`profile-${prefix}-age`).textContent = stats.age ? `Age ${stats.age}` : "";

        const grid = document.getElementById(`stats-${prefix}`);
        grid.innerHTML = "";

        const items = [
            ["Height", stats.height],
            ["Reach", stats.reach],
            ["Weight", stats.weight ? stats.weight + " lbs" : null],
            ["Win Streak", stats.currentWinStreak > 0 ? stats.currentWinStreak : "\u2014"],
            ["Sig. Str/Min", fmt(stats.avgSigStrLanded)],
            ["Str. Acc.", fmt(stats.avgSigStrPct, "%")],
            ["TD Avg", fmt(stats.avgTDLanded)],
            ["TD Acc.", fmt(stats.avgTDPct, "%")],
            ["Sub Avg", fmt(stats.avgSubAtt)],
            ["Title Bouts", fmt(stats.titleBouts)],
            ["Total Rnds", fmt(stats.totalRounds)],
            ["Best Streak", fmt(stats.longestWinStreak)],
        ];

        items.forEach(([label, value]) => {
            const div = document.createElement("div");
            div.className = "stat-item";
            div.innerHTML = `<span class="stat-label">${esc(label)}</span><span class="stat-value">${esc(value || "\u2014")}</span>`;
            grid.appendChild(div);
        });
    }

    // ---- Head-to-Head Comparison ----
    function renderComparison(name1, name2, f1, f2) {
        const rows = document.getElementById("comparison-rows");
        rows.innerHTML = "";

        const metrics = [
            ["Wins", f1.wins, f2.wins],
            ["Losses", f1.losses, f2.losses, true],
            ["Sig. Str/Min", f1.avgSigStrLanded, f2.avgSigStrLanded],
            ["Str. Accuracy", f1.avgSigStrPct, f2.avgSigStrPct],
            ["TD Average", f1.avgTDLanded, f2.avgTDLanded],
            ["TD Accuracy", f1.avgTDPct, f2.avgTDPct],
            ["Sub Attempts", f1.avgSubAtt, f2.avgSubAtt],
            ["Win Streak", f1.currentWinStreak, f2.currentWinStreak],
        ];

        metrics.forEach(([label, v1, v2, invertAdvantage]) => {
            const val1 = v1 ?? 0;
            const val2 = v2 ?? 0;
            const total = val1 + val2 || 1;
            const pct1 = (val1 / total) * 100;
            const pct2 = (val2 / total) * 100;
            const leftLeads = invertAdvantage ? val1 < val2 : val1 > val2;
            const rightLeads = invertAdvantage ? val2 < val1 : val2 > val1;

            const row = document.createElement("div");
            row.className = "cmp-row";
            row.innerHTML = `
                <div class="cmp-val left ${leftLeads ? 'advantage' : ''}">${fmt(v1)}</div>
                <div class="cmp-center">
                    <div class="cmp-label">${esc(label)}</div>
                    <div class="cmp-bar-wrap">
                        <div class="cmp-bar-left ${leftLeads ? 'lead' : ''}" style="width:${pct1}%"></div>
                        <div class="cmp-bar-right ${rightLeads ? 'lead' : ''}" style="width:${pct2}%"></div>
                    </div>
                </div>
                <div class="cmp-val right ${rightLeads ? 'advantage' : ''}">${fmt(v2)}</div>
            `;
            rows.appendChild(row);
        });
    }

    // ---- Win Methods ----
    function renderMethods(f1, f2) {
        const methods = [
            ["KO/TKO", f1.koWins, f2.koWins],
            ["Submission", f1.subWins, f2.subWins],
            ["Decision", f1.decWins, f2.decWins],
        ];
        const maxVal = Math.max(...methods.flatMap(([, a, b]) => [a || 0, b || 0]), 1);

        const colF1 = document.getElementById("method-f1");
        const colF2 = document.getElementById("method-f2");
        const labels = document.getElementById("method-labels");
        colF1.innerHTML = "";
        colF2.innerHTML = "";
        labels.innerHTML = "";

        methods.forEach(([label, v1, v2]) => {
            const w1 = Math.max(((v1 || 0) / maxVal) * 100, 3);
            const w2 = Math.max(((v2 || 0) / maxVal) * 100, 3);
            labels.innerHTML += `<div class="method-row-label">${esc(label)}</div>`;
            colF1.innerHTML += `
                <div class="method-bar-wrap left">
                    <div class="method-bar red-bar" style="width:${w1}%">
                        <span class="method-count">${v1 || 0}</span>
                    </div>
                </div>`;
            colF2.innerHTML += `
                <div class="method-bar-wrap right">
                    <div class="method-bar blue-bar" style="width:${w2}%">
                        <span class="method-count">${v2 || 0}</span>
                    </div>
                </div>`;
        });
    }

    // ---- Key Factors (NEW) ----
    const CATEGORY_COLORS = {
        striking: "#ef4444",
        grappling: "#22c55e",
        physical: "#a855f7",
        experience: "#f59e0b",
        other: "#6b7280",
    };

    function renderKeyFactors(factors, f1Name, f2Name) {
        const container = document.getElementById("key-factors");
        container.innerHTML = "";
        if (!factors || !factors.length) return;

        const maxImpact = Math.max(...factors.map(f => f.impact));

        factors.forEach(factor => {
            const isF1 = factor.advantage === f1Name;
            const barPct = Math.max((factor.impact / maxImpact) * 100, 8);
            const catColor = CATEGORY_COLORS[factor.category] || CATEGORY_COLORS.other;

            const row = document.createElement("div");
            row.className = "factor-row";
            row.innerHTML = `
                <div class="factor-top">
                    <span class="factor-name">${esc(factor.factor)}</span>
                    <span class="factor-cat" style="background: ${catColor}20; color: ${catColor}; border-color: ${catColor}40">${esc(factor.category)}</span>
                </div>
                <div class="factor-bottom">
                    <span class="factor-val ${isF1 ? 'red-text' : ''}">${esc(String(factor.f1Value))}</span>
                    <div class="factor-bar-wrap">
                        <div class="factor-bar ${isF1 ? 'red-fill' : 'blue-fill'}" style="width:${barPct}%"></div>
                    </div>
                    <span class="factor-val ${!isF1 ? 'blue-text' : ''}">${esc(String(factor.f2Value))}</span>
                </div>
                <div class="factor-adv ${isF1 ? 'red-text' : 'blue-text'}">
                    ${isF1 ? '\u25C0' : '\u25B6'} Favors ${esc(factor.advantage)}
                </div>
            `;
            container.appendChild(row);
        });
    }

    // ---- Category Edge Analysis (NEW) ----
    function renderCategoryEdges(categories, f1Name, f2Name) {
        const container = document.getElementById("category-edges");
        container.innerHTML = "";
        if (!categories) return;

        document.getElementById("edge-f1-name").textContent = f1Name;
        document.getElementById("edge-f2-name").textContent = f2Name;

        const order = ["striking", "grappling", "physical", "experience"];

        order.forEach(key => {
            const cat = categories[key];
            if (!cat) return;

            const score = cat.score;
            const f1Pct = score;
            const f2Pct = 100 - score;
            const catColor = CATEGORY_COLORS[key] || CATEGORY_COLORS.other;

            const row = document.createElement("div");
            row.className = "edge-row";
            row.innerHTML = `
                <div class="edge-label">
                    <span class="edge-dot" style="background: ${catColor}"></span>
                    <span class="edge-name">${esc(cat.label)}</span>
                </div>
                <div class="edge-bar-container">
                    <div class="edge-bar-track">
                        <div class="edge-fill-left" style="width:${f1Pct}%"></div>
                        <div class="edge-fill-right" style="width:${f2Pct}%"></div>
                    </div>
                    <div class="edge-midline"></div>
                </div>
                <div class="edge-score ${score > 55 ? 'red-text' : score < 45 ? 'blue-text' : ''}">${score}</div>
            `;
            container.appendChild(row);
        });
    }

    // ---- Model Breakdown (NEW) ----
    const MODEL_DISPLAY = {
        logistic_regression: "Logistic Regression",
        random_forest: "Random Forest",
        gradient_boosting: "Gradient Boosting",
        ensemble: "Ensemble",
    };

    function renderModelBreakdown(breakdown, f1Name, f2Name) {
        const container = document.getElementById("model-breakdown");
        container.innerHTML = "";
        if (!breakdown) return;

        document.getElementById("models-f1-name").textContent = f1Name;
        document.getElementById("models-f2-name").textContent = f2Name;

        const order = ["logistic_regression", "random_forest", "gradient_boosting", "ensemble"];

        order.forEach(key => {
            const model = breakdown[key];
            if (!model) return;

            const f1Pct = (model.f1Prob * 100).toFixed(1);
            const f2Pct = (model.f2Prob * 100).toFixed(1);
            const isEnsemble = key === "ensemble";
            const accText = model.accuracy ? `${(model.accuracy * 100).toFixed(1)}% acc` : "";

            const row = document.createElement("div");
            row.className = `model-row ${isEnsemble ? "model-ensemble" : ""}`;
            row.innerHTML = `
                <div class="model-info">
                    <span class="model-name">${esc(MODEL_DISPLAY[key] || key)}</span>
                    ${accText ? `<span class="model-acc">${accText}</span>` : ""}
                </div>
                <div class="model-bar-wrap">
                    <div class="model-bar">
                        <div class="model-fill red-fill" style="width:${f1Pct}%"></div>
                        <div class="model-fill blue-fill" style="width:${f2Pct}%"></div>
                    </div>
                </div>
                <div class="model-probs">
                    <span class="model-prob red-text">${f1Pct}%</span>
                    <span class="model-prob blue-text">${f2Pct}%</span>
                </div>
            `;
            container.appendChild(row);
        });
    }

    // ---- Historical Matchups (NEW) ----
    function renderHistorical(matchups, f1Name, f2Name) {
        const card = document.getElementById("history-card");
        const container = document.getElementById("historical-matchups");
        container.innerHTML = "";

        if (!matchups || !matchups.length) {
            card.classList.add("hidden");
            return;
        }

        card.classList.remove("hidden");

        matchups.forEach(m => {
            const dateStr = esc(m.date || "Unknown date");
            const methodParts = [m.method];
            if (m.round) methodParts.push(`Round ${m.round}`);
            if (m.time) methodParts.push(m.time);

            const isF1Winner = m.winner === f1Name;
            const isF2Winner = m.winner === f2Name;

            const row = document.createElement("div");
            row.className = "history-row";
            row.innerHTML = `
                <div class="history-date">${dateStr}</div>
                <div class="history-result">
                    <span class="history-winner ${isF1Winner ? 'red-text' : isF2Winner ? 'blue-text' : ''}">${m.winner}</span>
                    <span class="history-method">${methodParts.filter(Boolean).join(" \u00B7 ")}</span>
                </div>
            `;
            container.appendChild(row);
        });
    }
});
