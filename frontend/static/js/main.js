/**
 * AquaPulse — Main JavaScript
 * Handles navigation effects, rain animation, fetch API calls,
 * smooth scroll, and general UI enhancements.
 */

"use strict";

/* ─── Utility ────────────────────────────────────────────────── */
const qs  = (sel, ctx = document) => ctx.querySelector(sel);
const qsa = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

/* ─── Navbar scroll effect ───────────────────────────────────── */
function initNavbar() {
  const navbar = qs(".navbar");
  if (!navbar) return;
  const handler = () => {
    navbar.style.boxShadow = window.scrollY > 20
      ? "0 2px 24px rgba(0,0,0,0.5)"
      : "none";
  };
  window.addEventListener("scroll", handler, { passive: true });
}

/* ─── Rain Canvas Animation ──────────────────────────────────── */
function initRainCanvas() {
  const canvas = qs("#rainCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let drops = [];
  let animId;

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    spawnDrops();
  }

  function spawnDrops() {
    const count = Math.floor(canvas.width / 8);
    drops = Array.from({ length: count }, () => ({
      x:     Math.random() * canvas.width,
      y:     Math.random() * canvas.height,
      len:   Math.random() * 20 + 10,
      speed: Math.random() * 4 + 2,
      alpha: Math.random() * 0.4 + 0.1,
      width: Math.random() * 1 + 0.4,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drops.forEach(d => {
      ctx.beginPath();
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(d.x - 1, d.y + d.len);
      ctx.strokeStyle = `rgba(0, 212, 255, ${d.alpha})`;
      ctx.lineWidth = d.width;
      ctx.stroke();

      d.y += d.speed;
      if (d.y - d.len > canvas.height) {
        d.y = -d.len;
        d.x = Math.random() * canvas.width;
      }
    });
    animId = requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener("resize", resize, { passive: true });

  // Stop animation when hero is not visible (perf)
  const hero = qs(".hero");
  if (hero && "IntersectionObserver" in window) {
    const obs = new IntersectionObserver(entries => {
      entries[0].isIntersecting ? requestAnimationFrame(draw) : cancelAnimationFrame(animId);
    });
    obs.observe(hero);
  }
}

/* ─── Counter Animation ──────────────────────────────────────── */
function initCounters() {
  qsa(".counter").forEach(el => {
    const target = parseInt(el.dataset.target || el.textContent, 10);
    if (isNaN(target)) return;
    let start = 0;
    const duration = 1200;
    const startTime = performance.now();

    const update = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(start + (target - start) * eased);
      if (progress < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  });
}

/* ─── Intersection Observer for section reveal ───────────────── */
function initReveal() {
  if (!("IntersectionObserver" in window)) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("revealed");
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  qsa(".stat-card, .feature-card, .pipeline-step, .kpi-card, .insight-card, .tech-card").forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(20px)";
    el.style.transition = "opacity 0.4s ease, transform 0.4s ease";
    obs.observe(el);
  });

  // Add revealed styles
  const style = document.createElement("style");
  style.textContent = ".revealed { opacity: 1 !important; transform: translateY(0) !important; }";
  document.head.appendChild(style);
}

/* ─── Gallery image modal / lightbox ────────────────────────── */
function initGallery() {
  const cards = qsa(".gallery-card");
  if (!cards.length) return;

  // Create modal
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    display:none; position:fixed; inset:0; z-index:9999;
    background:rgba(0,0,0,0.88); backdrop-filter:blur(8px);
    align-items:center; justify-content:center; padding:2rem;
    cursor:zoom-out;
  `;
  const img = document.createElement("img");
  img.style.cssText = "max-width:100%; max-height:90vh; border-radius:12px; box-shadow:0 20px 60px rgba(0,0,0,0.6);";
  const caption = document.createElement("div");
  caption.style.cssText = "position:absolute; bottom:1.5rem; color:rgba(255,255,255,0.6); font-size:0.8rem; font-family:Inter,sans-serif;";
  overlay.appendChild(img);
  overlay.appendChild(caption);
  document.body.appendChild(overlay);

  cards.forEach(card => {
    card.addEventListener("click", () => {
      const src = qs("img", card)?.src;
      const lbl = qs(".gallery-label", card)?.textContent;
      if (!src) return;
      img.src = src;
      caption.textContent = lbl || "";
      overlay.style.display = "flex";
    });
  });

  overlay.addEventListener("click", () => { overlay.style.display = "none"; });
  document.addEventListener("keydown", e => { if (e.key === "Escape") overlay.style.display = "none"; });
}

/* ─── Smooth active nav on scroll ───────────────────────────── */
function initSmoothScroll() {
  qsa('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      const id = a.getAttribute("href").slice(1);
      const target = document.getElementById(id);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

/* ─── Toast notification ─────────────────────────────────────── */
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  const colors = { info: "#00d4ff", success: "#00b894", error: "#ff6b6b", warn: "#fdcb6e" };
  toast.style.cssText = `
    position:fixed; bottom:1.5rem; right:1.5rem; z-index:9998;
    background:#0f2544; color:#f0f4f8; border-left:3px solid ${colors[type] || colors.info};
    padding:0.85rem 1.25rem; border-radius:10px; font-size:0.85rem;
    font-family:Inter,sans-serif; box-shadow:0 8px 32px rgba(0,0,0,0.4);
    animation:slideInToast 0.3s ease;
    max-width:320px; line-height:1.4;
  `;
  toast.textContent = message;

  const anim = document.createElement("style");
  anim.textContent = `
    @keyframes slideInToast {
      from { opacity:0; transform:translateY(16px); }
      to   { opacity:1; transform:translateY(0); }
    }
  `;
  document.head.appendChild(anim);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

/* ─── API helpers ────────────────────────────────────────────── */
async function apiFetch(url, options = {}) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(`[AquaPulse] API error (${url}):`, err);
    throw err;
  }
}

/* ─── Prediction form helper (standalone page) ───────────────── */
function initPredictionPageHelpers() {
  const form = qs("#predictionForm");
  if (!form) return;

  const btn = qs("#predictBtn");
  const btnText = qs("#btnText");

  form.addEventListener("submit", () => {
    if (btn) btn.disabled = true;
    if (btnText) btnText.textContent = "Running…";
    setTimeout(() => {
      if (btn) btn.disabled = false;
      if (btnText) btnText.textContent = "Generate Prediction";
    }, 5000);
  });
}

/* ─── Init ───────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initRainCanvas();
  initReveal();
  initCounters();
  initGallery();
  initSmoothScroll();
  initPredictionPageHelpers();
});

/* ─── Export for inline scripts ──────────────────────────────── */
window.AquaPulse = { apiFetch, showToast };
