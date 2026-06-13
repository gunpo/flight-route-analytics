/* ============================================================
   Flight Route Efficiency — front-end logic
   Loads data.json, renders Chart.js charts, and runs the
   logistic-regression model client-side for the predictor.
   ============================================================ */

const FONT_MONO = "IBM Plex Mono, monospace";
const C = {
  amber: "#FFB02E", cyan: "#5BD1E8", coral: "#FF6F61",
  ink: "#EAF0F6", muted: "#8298B0", line: "#274058", navy3: "#1E3350",
};
const MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

let DATA = null;

init();

async function init() {
  try {
    const res = await fetch("data.json");
    if (!res.ok) throw new Error(res.status);
    DATA = await res.json();
  } catch (e) {
    document.querySelector(".hero-sub").textContent =
      "Could not load data.json — run `python src/build_site.py` first.";
    console.error(e);
    return;
  }
  fillMeta();
  const steps = [drawBoard, drawCascade, drawLeaderboard, drawCarrier,
                 drawSeason, buildPredictor, setRepoLink];
  steps.forEach((fn) => {
    try { fn(); } catch (e) { console.error(`${fn.name} failed:`, e); }
  });
}

/* ---------- meta text ---------- */
function fillMeta() {
  const m = DATA.meta;
  const fmt = {
    total_flights: m.total_flights.toLocaleString(),
    n_routes: m.n_routes,
    n_carriers: m.n_carriers,
    pct_delayed: m.pct_delayed,
    best_hour: m.best_hour,
    date_range: m.date_start.slice(0, 4),
  };
  document.querySelectorAll("[data-meta]").forEach((el) => {
    const key = el.dataset.meta;
    if (key in fmt && !el.hasAttribute("data-flap")) el.textContent = fmt[key];
  });
  document.querySelectorAll("[data-model]").forEach((el) => {
    const k = el.dataset.model;
    const v = DATA.model.metrics[k];
    el.textContent = k === "n_train" ? v.toLocaleString() : v;
  });
}

/* ---------- split-flap count-up ---------- */
function drawBoard() {
  const cells = document.querySelectorAll("[data-flap]");
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const el = en.target;
      const key = el.dataset.meta;
      const target = DATA.meta[key];
      const suffix = el.dataset.suffix || "";
      countUp(el, target, suffix);
      obs.unobserve(el);
    });
  }, { threshold: 0.5 });
  cells.forEach((c) => obs.observe(c));
}

function countUp(el, target, suffix) {
  const dur = 1100, start = performance.now();
  const isFloat = !Number.isInteger(target);
  function frame(now) {
    const t = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = target * eased;
    el.textContent = (isFloat ? val.toFixed(1) : Math.round(val).toLocaleString()) + suffix;
    if (t < 1) requestAnimationFrame(frame);
    else el.textContent = (isFloat ? target.toFixed(1) : target.toLocaleString()) + suffix;
  }
  requestAnimationFrame(frame);
}

/* ---------- shared chart defaults ---------- */
function baseOpts(yTitle) {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#0E1A2B", borderColor: C.line, borderWidth: 1,
        titleFont: { family: FONT_MONO }, bodyFont: { family: FONT_MONO },
        padding: 10,
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(39,64,88,0.5)" },
        ticks: { color: C.muted, font: { family: FONT_MONO, size: 11 } },
      },
      y: {
        grid: { color: "rgba(39,64,88,0.5)" },
        ticks: { color: C.muted, font: { family: FONT_MONO, size: 11 } },
        title: { display: !!yTitle, text: yTitle, color: C.muted,
                 font: { family: FONT_MONO, size: 11 } },
      },
    },
  };
}

/* ---------- cascade ---------- */
function drawCascade() {
  const m = DATA.meta;
  const best = DATA.hourly.find((d) => d.hour === m.best_hour);
  const worst = DATA.hourly.find((d) => d.hour === m.worst_hour);
  const mult = (worst.avg_dep_delay / Math.max(best.avg_dep_delay, 0.1)).toFixed(1);
  document.getElementById("cascadeTakeaway").textContent =
    `A ${m.worst_hour}:00 departure averages ${worst.avg_dep_delay} min late — ` +
    `${mult}× the ${best.avg_dep_delay} min of a ${m.best_hour}:00 flight.`;

  const h = DATA.hourly;
  const labels = h.map((d) => d.hour + ":00");
  const ctx = document.getElementById("cascadeChart");
  const grad = ctx.getContext("2d").createLinearGradient(0, 0, 0, 360);
  grad.addColorStop(0, "rgba(255,176,46,0.35)");
  grad.addColorStop(1, "rgba(255,176,46,0)");

  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: h.map((d) => d.avg_dep_delay),
        borderColor: C.amber, backgroundColor: grad,
        borderWidth: 2.5, fill: true, tension: 0.35,
        pointRadius: 0, pointHoverRadius: 5, pointHoverBackgroundColor: C.amber,
      }],
    },
    options: { ...baseOpts("Avg departure delay (min)") },
  });
}

/* ---------- leaderboard ---------- */
function drawLeaderboard() {
  const routes = DATA.routes;
  const top = routes.slice(0, 6);
  const bottom = routes.slice(-4);
  const show = [...top, ...bottom];
  const max = Math.max(...routes.map((r) => r.score));
  const wrap = document.getElementById("leaderboard");

  show.forEach((r, i) => {
    const rank = routes.indexOf(r) + 1;
    const color = r.score >= 70 ? C.cyan : r.score >= 55 ? C.amber : C.coral;
    const row = document.createElement("div");
    row.className = "lb-row";
    row.innerHTML =
      `<span class="lb-rank">${String(rank).padStart(2, "0")}</span>` +
      `<span class="lb-route">${r.route}</span>` +
      `<span class="lb-bar-track"><span class="lb-bar" style="width:0%;background:${color};color:${color}"></span></span>` +
      `<span class="lb-score" style="color:${color}">${r.score}</span>`;
    wrap.appendChild(row);
    if (i === 6) {
      const div = document.createElement("p");
      div.style.cssText = "font-family:var(--mono);color:var(--faint);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;margin:8px 0 2px";
      div.textContent = "— the bottom of the board —";
      wrap.insertBefore(div, row);
    }
    requestAnimationFrame(() =>
      setTimeout(() => { row.querySelector(".lb-bar").style.width = (r.score / max * 100) + "%"; }, 100 + i * 70));
  });
}

/* ---------- carrier recovery ---------- */
function drawCarrier() {
  const c = [...DATA.carriers].sort((a, b) => b.avg_min_recovered - a.avg_min_recovered);
  new Chart(document.getElementById("carrierChart"), {
    type: "bar",
    data: {
      labels: c.map((d) => d.name),
      datasets: [{
        data: c.map((d) => d.avg_min_recovered),
        backgroundColor: c.map((_, i) => i === 0 ? C.cyan : C.navy3),
        borderColor: c.map((_, i) => i === 0 ? C.cyan : C.line),
        borderWidth: 1, borderRadius: 5,
      }],
    },
    options: { ...baseOpts("Min recovered") },
  });
}

/* ---------- seasonality ---------- */
function drawSeason() {
  const m = DATA.monthly;
  new Chart(document.getElementById("seasonChart"), {
    type: "line",
    data: {
      labels: m.map((d) => MONTHS[d.month]),
      datasets: [{
        data: m.map((d) => d.avg_arr_delay),
        borderColor: C.coral, backgroundColor: "rgba(255,111,97,0.12)",
        borderWidth: 2.5, fill: true, tension: 0.35,
        pointRadius: 0, pointHoverRadius: 5,
      }],
    },
    options: { ...baseOpts("Avg arrival delay (min)") },
  });
}

/* ============================================================
   PREDICTOR — logistic regression inference in the browser
   ============================================================ */
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function predict(route, carrier, hour, month) {
  const M = DATA.model;
  // Reconstruct the training feature vector in the exact same order.
  const raw = {
    DEP_HOUR: hour,
    DISTANCE: route.distance,
    ROUTE_CONGESTION: route.congestion,
    MONTH: month,
  };
  // one-hot carriers (baseline carrier = first in list, dropped)
  M.carriers.slice(1).forEach((code) => {
    raw["CARR_" + code] = carrier === code ? 1 : 0;
  });

  let z = M.intercept;
  M.features.forEach((f, i) => {
    const standardized = (raw[f] - M.means[i]) / M.scales[i];
    z += M.coef[i] * standardized;
  });
  return sigmoid(z);
}

function buildPredictor() {
  const selRoute = document.getElementById("selRoute");
  const selCarrier = document.getElementById("selCarrier");
  const selHour = document.getElementById("selHour");
  const selMonth = document.getElementById("selMonth");

  // routes sorted alphabetically for usability
  [...DATA.routes].sort((a, b) => a.route.localeCompare(b.route)).forEach((r) => {
    const o = document.createElement("option");
    o.value = r.route; o.textContent = `${r.route}  ·  ${r.distance} mi`;
    selRoute.appendChild(o);
  });
  DATA.carriers.forEach((c) => {
    const o = document.createElement("option");
    o.value = c.code; o.textContent = c.name;
    selCarrier.appendChild(o);
  });
  DATA.hourly.forEach((h) => {
    const o = document.createElement("option");
    o.value = h.hour; o.textContent = String(h.hour).padStart(2, "0") + ":00";
    selHour.appendChild(o);
  });
  for (let i = 1; i <= 12; i++) {
    const o = document.createElement("option");
    o.value = i; o.textContent = MONTHS[i];
    selMonth.appendChild(o);
  }
  // sensible defaults: a notoriously bad combo so the demo shows range
  selRoute.value = DATA.routes[DATA.routes.length - 1].route;
  selHour.value = DATA.meta.worst_hour;
  selMonth.value = 7;

  [selRoute, selCarrier, selHour, selMonth].forEach((s) =>
    s.addEventListener("change", update));
  update();

  function update() {
    const route = DATA.routes.find((r) => r.route === selRoute.value);
    const p = predict(route, selCarrier.value,
                       +selHour.value, +selMonth.value);
    renderGauge(p);
  }
}

function renderGauge(p) {
  const pct = Math.round(p * 100);
  const fill = document.getElementById("gaugeFill");
  const len = 251.3;
  fill.style.strokeDashoffset = len * (1 - p);

  const color = p < 0.35 ? C.cyan : p < 0.6 ? C.amber : C.coral;
  fill.style.stroke = color;

  const pctEl = document.getElementById("gaugePct");
  pctEl.textContent = pct + "%";
  pctEl.style.color = color;

  const verdict = document.getElementById("verdict");
  verdict.style.color = color;
  verdict.textContent =
    p < 0.35 ? "Low risk — book it." :
    p < 0.6  ? "Coin flip. Have a backup plan." :
               "High risk. Expect to wait.";
}

/* ---------- repo link placeholder ---------- */
function setRepoLink() {
  // Your GitHub repository.
  const USER = "gunpo";
  const REPO = "flight-route-analytics";
  const link = document.getElementById("repoLink");
  if (USER !== "YOUR_GITHUB_USERNAME")
    link.href = `https://github.com/${USER}/${REPO}`;
  else link.removeAttribute("href"), link.style.opacity = 0.5;
}
