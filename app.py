"""
app.py — ZAPDOS LABS · Autonomous Floor Command (revamped UI)

Run:  python3 -m streamlit run app.py
"""

import html
import os
import time
import base64
import tempfile
import datetime

import streamlit as st

from flask import Flask

# This MUST be at the top level! Vercel is looking for this exact variable.
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World!'

try:
    import cv2
except ImportError:
    cv2 = None

import video as videolib
from vlm import VLMError
from agents import run_shift, ROUTE_MAP
from run import discover_sequences

try:
    import detect as detlib
except ImportError:
    detlib = None

MODEL = os.environ.get("VLLM_MODEL", "(set VLLM_MODEL)")
MAX_PER_SEQ = 4
MAX_SEQS = 5

AGENTS = [
    {"key": "supervisor", "icon": "🎯", "name": "Shift Supervisor", "role": "Orchestrator",
     "accent": "#6366f1", "bg": "#eef2ff", "default": "Watching the floor feed…"},
    {"key": "safety", "icon": "🦺", "name": "Safety Officer", "role": "SafetyCulture",
     "accent": "#ea580c", "bg": "#fff7ed", "default": "On standby for EHS hazards."},
    {"key": "quality", "icon": "🔬", "name": "Quality Inspector", "role": "MasterControl",
     "accent": "#9333ea", "bg": "#faf5ff", "default": "Ready to inspect batches."},
    {"key": "inventory", "icon": "📦", "name": "Inventory Clerk", "role": "Manhattan WMS",
     "accent": "#059669", "bg": "#ecfdf5", "default": "Tracking stock on the floor."},
    {"key": "maintenance", "icon": "🔧", "name": "Maintenance Tech", "role": "Maximo",
     "accent": "#2563eb", "bg": "#eff6ff", "default": "Monitoring equipment health."},
    {"key": "dispatch", "icon": "🚛", "name": "Floor Dispatcher", "role": "Dispatch TMS",
     "accent": "#d97706", "bg": "#fffbeb", "default": "Coordinating vehicle flow."},
]

ROUTE_BGR = {
    "safety": (60, 146, 251),
    "quality": (250, 139, 167),
    "inventory": (153, 211, 52),
    "maintenance": (250, 165, 96),
    "dispatch": (36, 191, 251),
}

SYSTEM_LABEL = {
    "safety": "SafetyCulture",
    "quality": "MasterControl",
    "inventory": "Manhattan WMS",
    "maintenance": "Maximo",
    "dispatch": "Dispatch TMS",
}

DEFAULT_MSG = {a["key"]: a["default"] for a in AGENTS}

SEV_RANK = {"CRITICAL": 4, "HIGH": 3, "MED": 2, "LOW": 1, "HELD": 2, "OK": 0}
SEV_COLOR = {
    "CRITICAL": "#dc2626", "HIGH": "#ea580c", "MED": "#d97706",
    "LOW": "#2563eb", "HELD": "#94a3b8", "OK": "#10b981",
}


class AlertFeed:
    """Ranked floor alerts only — no procedural noise."""

    def __init__(self):
        self._items = []

    def add(self, severity, text, moment=None):
        sev = str(severity or "MED").upper()
        if sev not in SEV_RANK:
            sev = "MED"
        if sev == "OK" and SEV_RANK["OK"] == 0:
            return  # skip noise
        rank = SEV_RANK[sev]
        label = ("M%d · " % moment) if moment else ""
        self._items.append({
            "rank": rank, "severity": sev,
            "html": label + str(text),
            "color": SEV_COLOR.get(sev, "#6366f1"),
        })
        self._items.sort(key=lambda x: (-x["rank"], x["html"]))

    def render(self):
        if not self._items:
            return ('<div class="alert-feed"><div class="alert-empty">'
                    'No issues flagged yet.</div></div>')
        rows = ['<div class="alert-feed">']
        for it in self._items:
            rows.append(
                '<div class="alert-line" style="border-left-color:%s">'
                '<span class="rank rank-%s">%s</span>'
                '<span class="txt">%s</span></div>'
                % (it["color"], it["severity"], it["severity"], _esc(it["html"])))
        rows.append('</div>')
        return "".join(rows)

st.set_page_config(page_title="Zapdos · Floor Command", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp {
  background: linear-gradient(165deg, #f8fafc 0%, #eef2ff 45%, #f0fdf4 100%);
  color: #0f172a;
}
#MainMenu, header, footer { visibility: hidden; }
section.main > div { padding-top: 0.25rem; max-width: 100%; margin: 0 auto; }
.block-container { padding-top: 0.5rem; padding-bottom: 1rem; max-width: 100%; }

/* Streamlit widgets */
[data-testid="stFileUploader"] {
  background: #fff; border: 2px dashed #c7d2fe; border-radius: 16px;
  padding: 8px 12px; transition: border-color .2s, box-shadow .2s;
}
[data-testid="stFileUploader"]:hover {
  border-color: #818cf8; box-shadow: 0 4px 20px rgba(99,102,241,.12);
}
[data-testid="stFileUploader"] label { font-weight: 600 !important; color: #334155 !important; }
[data-testid="stFileUploader"] small { color: #64748b !important; }
div[data-testid="stSelectbox"] > div { border-radius: 12px; border-color: #e2e8f0; }
.stButton > button {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%) !important;
  color: #fff !important; border: none !important; border-radius: 14px !important;
  padding: 0.85rem 1.5rem !important; font-size: 1.05rem !important; font-weight: 800 !important;
  letter-spacing: 0.02em; box-shadow: 0 8px 24px rgba(99,102,241,.35) !important;
  transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 32px rgba(99,102,241,.45) !important;
}
[data-testid="stToggle"] label { font-weight: 600; color: #475569; }

/* Video column panel */
.vid-panel {
  background: #fff; border-radius: 14px; border: 1px solid #e2e8f0;
  padding: 10px 12px; box-shadow: 0 2px 10px rgba(15,23,42,.04);
}
[data-testid="stVideo"] { border-radius: 12px; overflow: hidden; }
[data-testid="stVideo"] video {
  max-height: 260px; width: 100%; object-fit: contain;
  background: #0f172a; border-radius: 12px;
}
.video-shell {
  background: #0f172a; border-radius: 14px; overflow: hidden;
  border: 1px solid #e2e8f0; min-height: 48px;
}
.video-placeholder {
  padding: 28px 16px; text-align: center; color: #94a3b8;
  font-size: 0.85rem; background: #f8fafc; border-radius: 14px;
  border: 1px dashed #cbd5e1;
}

/* Header */
.site-header {
  display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
  padding: 12px 18px; background: #fff; border-radius: 14px;
  border: 1px solid #e2e8f0; box-shadow: 0 2px 12px rgba(15,23,42,.04);
}
.site-header .bolt { font-size: 28px; }
.site-header h1 {
  margin: 0; font-size: 1.25rem; font-weight: 800; color: #0f172a;
  background: linear-gradient(90deg, #4f46e5, #7c3aed);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.site-header .eyebrow {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
  color: #64748b; text-transform: uppercase;
}
.site-header .tagline { font-size: 0.9rem; color: #64748b; margin-top: 2px; }
.live-pill {
  margin-left: auto; display: flex; align-items: center; gap: 8px;
  background: #ecfdf5; color: #047857; font-weight: 700; font-size: 0.8rem;
  padding: 8px 14px; border-radius: 999px; border: 1px solid #a7f3d0;
}
.live-pill .dot {
  width: 8px; height: 8px; background: #10b981; border-radius: 50%;
  animation: pulse-dot 1.4s ease infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.2)} }

/* Upload bar */
.upload-bar {
  margin-bottom: 10px; padding: 10px 14px; background: #fff; border-radius: 14px;
  border: 1px solid #e2e8f0; box-shadow: 0 2px 10px rgba(15,23,42,.03);
}

/* Agent crew — compact 6-across */
.crew-title {
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em;
  text-transform: uppercase; color: #64748b; margin: 0 0 6px 2px;
}
.agent-crew {
  display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px;
  margin-bottom: 10px;
}
@media (max-width: 1200px) { .agent-crew { grid-template-columns: repeat(3, 1fr); } }

.agent-tile {
  background: #fff; border-radius: 12px; padding: 0;
  border: 2px solid #e2e8f0; overflow: hidden;
  box-shadow: 0 2px 8px rgba(15,23,42,.04);
  transition: border-color .2s, box-shadow .2s;
}
.agent-tile.thinking {
  border-color: #818cf8; box-shadow: 0 4px 16px rgba(99,102,241,.15);
}
.agent-tile.active {
  border-color: #10b981; box-shadow: 0 4px 16px rgba(16,185,129,.18);
}
.agent-tile.idle { opacity: 0.88; }

.bubble-wrap { padding: 8px 8px 0; min-height: 0; }
.speech-bubble {
  position: relative; padding: 8px 10px; border-radius: 10px;
  font-size: 0.78rem; line-height: 1.35; color: #1e293b; font-weight: 500;
  min-height: 40px; border: 1px solid rgba(0,0,0,.06);
  animation: pop-in .25s ease;
  word-wrap: break-word; overflow-wrap: break-word;
}
@keyframes pop-in { from{opacity:0;transform:scale(.96)} to{opacity:1;transform:scale(1)} }
.speech-bubble::after {
  content: ''; position: absolute; bottom: -9px; left: 28px;
  border: 9px solid transparent; border-top-color: inherit;
  border-top-width: 10px;
}
.speech-bubble .ts { font-size: 0.68rem; color: #94a3b8; font-weight: 600; margin-top: 6px; }

.agent-body {
  display: flex; align-items: center; gap: 8px; padding: 8px 10px 10px;
}
.agent-avatar {
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem; flex-shrink: 0;
  border: 1px solid rgba(0,0,0,.06);
}
.agent-info .name { font-size: 0.78rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
.agent-info .role {
  font-size: 0.58rem; font-weight: 700; color: #64748b;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.status-chip {
  display: none;
}
.status-chip.idle { background: #f1f5f9; color: #94a3b8; }
.status-chip.thinking { background: #eef2ff; color: #4f46e5; animation: blink-chip 1s infinite; }
.status-chip.active { background: #d1fae5; color: #047857; }
@keyframes blink-chip { 0%,100%{opacity:1} 50%{opacity:.6} }

/* Panels */
.panel {
  background: #fff; border-radius: 14px; border: 1px solid #e2e8f0;
  padding: 10px 12px; box-shadow: 0 2px 10px rgba(15,23,42,.04);
  margin-bottom: 8px;
}
.panel-label {
  font-size: 0.65rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: #64748b; margin-bottom: 8px;
}
.strip { display: flex; gap: 6px; overflow-x: auto; padding: 2px 0; }
.strip .fr { flex: 0 0 auto; border-radius: 8px; overflow: hidden;
  border: 2px solid #e2e8f0; transition: .2s; }
.strip .fr img { height: 52px; display: block; }
.strip .fr.active { border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99,102,241,.35); }
.strip .fr.done { border-color: #86efac; opacity: .8; }

/* KPI strip — inline compact */
.kpi-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.kpi {
  flex: 1; min-width: 64px; text-align: center; padding: 6px 4px;
  background: #fff; border-radius: 10px; border: 1px solid #e2e8f0;
}
.kpi .v { font-size: 1.1rem; font-weight: 800; color: #0f172a; }
.kpi .l { font-size: 0.58rem; font-weight: 700; color: #94a3b8;
  text-transform: uppercase; letter-spacing: 0.04em; }

/* Side column — ranked alerts only */
.side-panel {
  background: #fff; border-radius: 14px; border: 1px solid #e2e8f0;
  padding: 10px 12px; box-shadow: 0 2px 10px rgba(15,23,42,.04);
}
.side-panel .panel-label { margin-bottom: 6px; }
.alert-feed {
  overflow-y: auto; font-size: 0.84rem; line-height: 1.35;
  max-height: calc(100vh - 420px); min-height: 120px;
}
.alert-line {
  display: flex; align-items: flex-start; gap: 8px;
  padding: 8px 10px; margin-bottom: 6px; border-radius: 10px;
  background: #f8fafc; border-left: 4px solid #cbd5e1;
}
.alert-line .rank {
  flex-shrink: 0; font-size: 0.62rem; font-weight: 800;
  padding: 2px 6px; border-radius: 5px; letter-spacing: 0.04em;
}
.rank-CRITICAL { background: #fee2e2; color: #991b1b; }
.rank-HIGH { background: #ffedd5; color: #c2410c; }
.rank-MED { background: #fef3c7; color: #b45309; }
.rank-LOW { background: #dbeafe; color: #1d4ed8; }
.rank-HELD { background: #f1f5f9; color: #64748b; }
.rank-OK { background: #d1fae5; color: #047857; }
.alert-line .txt { color: #334155; flex: 1; }
.alert-empty { color: #94a3b8; font-size: 0.85rem; padding: 8px 4px; }
.writes-compact { max-height: 110px; overflow-y: auto; margin-top: 4px; }
.write-card {
  padding: 12px 14px; margin-bottom: 8px; border-radius: 12px;
  background: #f8fafc; border: 1px solid #e2e8f0;
  border-left: 4px solid #6366f1;
}
.write-card .wid { font-size: 0.68rem; font-weight: 700; color: #94a3b8; }
.write-card .wtitle { font-weight: 700; color: #0f172a; margin: 4px 0; }
.write-card .wmeta { font-size: 0.8rem; color: #64748b; }

/* Handoff dashboard */
.handoff-hero {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
  color: #fff; border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
}
.handoff-hero h2 { margin: 0 0 4px; font-size: 1.1rem; font-weight: 800; }
.handoff-hero p { margin: 0; opacity: .92; font-size: 0.88rem; }

.flowchart {
  display: flex; align-items: center; justify-content: center; flex-wrap: wrap;
  gap: 4px; padding: 10px; background: #fff; border-radius: 12px;
  border: 1px solid #e2e8f0; margin-bottom: 10px;
}
.flow-node {
  padding: 6px 10px; border-radius: 8px; font-weight: 700; font-size: 0.72rem;
  background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0;
}
.flow-node.done { background: #ecfdf5; color: #047857; border-color: #6ee7b7; }
.flow-node.highlight { background: #eef2ff; color: #4f46e5; border-color: #a5b4fc; }
.flow-arrow { font-size: 0.9rem; color: #cbd5e1; font-weight: 800; }

.summary-grid {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 10px;
}
.sum-card {
  background: #fff; border-radius: 10px; padding: 10px; text-align: center;
  border: 1px solid #e2e8f0;
}
.sum-card .num { font-size: 1.4rem; font-weight: 800; }
.sum-card .lbl { font-size: 0.62rem; font-weight: 700; color: #64748b;
  text-transform: uppercase; }

/* Moment cards — full observations, no truncation */
.moments-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 10px; }
.moment-card {
  background: #fff; border-radius: 12px; border: 1px solid #e2e8f0;
  padding: 10px 12px;
}
.moment-head {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 6px;
}
.moment-head .mn { font-weight: 800; font-size: 0.82rem; color: #0f172a; }
.moment-head .ma { font-size: 0.75rem; font-weight: 600; color: #6366f1; }
.moment-obs {
  font-size: 0.82rem; line-height: 1.45; color: #334155;
  white-space: normal; word-wrap: break-word;
}
.badge {
  display: inline-block; padding: 3px 8px; border-radius: 6px;
  font-size: 0.68rem; font-weight: 800; text-transform: uppercase;
}
.badge-ok { background: #d1fae5; color: #047857; }
.badge-hold { background: #fee2e2; color: #b91c1c; }
.badge-clear { background: #f1f5f9; color: #94a3b8; }
.badge-CRITICAL { background: #fee2e2; color: #991b1b; }
.badge-HIGH { background: #ffedd5; color: #c2410c; }
.badge-MED { background: #fef3c7; color: #b45309; }
.badge-LOW { background: #dbeafe; color: #1d4ed8; }

.records-section { margin-bottom: 8px; }
.records-section h3 {
  font-size: 0.72rem; font-weight: 800; color: #475569;
  text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 6px;
}
.record-row {
  display: flex; gap: 8px; align-items: flex-start; padding: 8px 10px;
  background: #fff; border-radius: 10px; border: 1px solid #e2e8f0;
  margin-bottom: 6px;
}
.record-row .ricon { font-size: 1.5rem; }
.record-row .rdetail { flex: 1; }
.record-row .rtitle { font-weight: 700; color: #0f172a; }
.record-row .rsub { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

.welcome-card {
  text-align: center; padding: 20px 16px; background: #fff;
  border-radius: 14px; border: 1px solid #e2e8f0; margin-top: 8px;
}
.welcome-card h2 { color: #0f172a; margin-bottom: 4px; font-size: 1rem; }
.welcome-card p { color: #64748b; font-size: 0.85rem; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)


def _esc(text):
    return html.escape(str(text or ""))


def _thumb_b64(path, h=72):
    if cv2 is None:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    img = cv2.imread(path)
    if img is None:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    h0, w0 = img.shape[:2]
    w = max(1, int(w0 * (h / h0)))
    img = cv2.resize(img, (w, h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def render_header():
    return (
        '<div class="site-header">'
        '<span class="bolt">⚡</span>'
        '<div><div class="eyebrow">Zapdos Labs · Hackathon 2026</div>'
        '<h1>Autonomous Floor Command</h1>'
        '<div class="tagline">Six agents · trained PPE YOLO + VLM · Watch → Decide → Act → Report</div></div>'
        '<div class="live-pill"><span class="dot"></span> VLM LIVE · %s</div>'
        '</div>' % _esc(MODEL.split("/")[-1][:32])
    )


def render_kpi_row(values):
    labels = ["Frames", "Safety", "Quality", "Inventory", "Maint", "Dispatch", "Autonomy"]
    cells = []
    for lb, val in zip(labels, values):
        cells.append('<div class="kpi"><div class="v">%s</div><div class="l">%s</div></div>'
                     % (_esc(val), lb))
    return '<div class="kpi-row">%s</div>' % "".join(cells)


def render_agent_crew(statuses, messages, timestamps=None):
    timestamps = timestamps or {}
    tiles = []
    for ag in AGENTS:
        key = ag["key"]
        stt = statuses.get(key, "idle")
        msg = messages.get(key, ag["default"])
        ts = timestamps.get(key, "")
        bubble_bg = ag["bg"]
        if stt == "thinking":
            bubble_bg = "#eef2ff"
        elif stt == "active":
            bubble_bg = "#ecfdf5"
        tiles.append(
            '<div class="agent-tile %s">'
            '<div class="bubble-wrap">'
            '<div class="speech-bubble" style="background:%s;border-color:%s">'
            '%s%s</div></div>'
            '<div class="agent-body">'
            '<div class="agent-avatar" style="background:%s">%s</div>'
            '<div class="agent-info"><div class="name">%s</div>'
            '<div class="role">%s</div></div>'
            '<span class="status-chip %s">%s</span>'
            '</div></div>'
            % (stt, bubble_bg, ag["accent"] + "33", _esc(msg),
               ('<div class="ts">%s</div>' % _esc(ts) if ts else ""),
               ag["bg"], ag["icon"], _esc(ag["name"]), _esc(ag["role"]),
               stt, stt))
    return '<div class="crew-title">Floor crew — live agent callouts</div><div class="agent-crew">%s</div>' % (
        "".join(tiles))


def render_strip(thumbs, active_idx=None, done=False):
    cells = []
    for i, b in enumerate(thumbs):
        cls = "fr"
        if done:
            cls += " done"
        elif active_idx is not None and i == active_idx:
            cls += " active"
        elif active_idx is not None and i < active_idx:
            cls += " done"
        cells.append('<div class="%s"><img src="data:image/jpeg;base64,%s"></div>' % (cls, b))
    return '<div class="strip">%s</div>' % "".join(cells)


def render_side_panel(alerts_html, writes_html):
    return (
        '<div class="side-panel">'
        '<div class="panel-label">🚨 Alerts <span style="font-weight:500;color:#94a3b8">'
        '(ranked)</span></div>%s'
        '<div class="panel-label" style="margin-top:10px">📋 Commits this shift</div>'
        '<div class="writes-compact">%s</div></div>'
        % (alerts_html, writes_html)
    )


def render_ticker(items):
    """Legacy — use AlertFeed.render() instead."""
    feed = AlertFeed()
    for it in items:
        feed.add("MED", it.get("html", ""))
    return feed.render()


def render_writes(incidents, ncrs, flags, wos, dispatches):
    html_parts = []
    for inc in reversed(incidents[-6:]):
        sev = inc.get("severity", "MED")
        html_parts.append(
            '<div class="write-card" style="border-left-color:#ea580c">'
            '<div class="wid">%s · SafetyCulture</div>'
            '<div class="wtitle">🦺 %s</div>'
            '<div class="wmeta">%s · <span class="badge badge-%s">%s</span></div></div>'
            % (_esc(inc.get("incident_id")), _esc(inc.get("hazard_type")),
               _esc(inc.get("location")), sev, sev))
    for ncr in reversed(ncrs[-4:]):
        sev = ncr.get("severity", "MED")
        html_parts.append(
            '<div class="write-card" style="border-left-color:#9333ea">'
            '<div class="wid">%s · MasterControl</div>'
            '<div class="wtitle">🔬 %s</div>'
            '<div class="wmeta">Batch %s · <span class="badge badge-%s">%s</span></div></div>'
            % (_esc(ncr.get("ncr_id")), _esc(ncr.get("defect_type")),
               _esc(ncr.get("batch_id")), sev, sev))
    for fl in reversed(flags[-4:]):
        html_parts.append(
            '<div class="write-card" style="border-left-color:#059669">'
            '<div class="wid">%s · WMS</div>'
            '<div class="wtitle">📦 %s</div>'
            '<div class="wmeta">%s</div></div>'
            % (_esc(fl.get("flag_id")), _esc(fl.get("issue_type")), _esc(fl.get("location"))))
    for wo in reversed(wos[-4:]):
        html_parts.append(
            '<div class="write-card" style="border-left-color:#2563eb">'
            '<div class="wid">%s · Maximo</div>'
            '<div class="wtitle">🔧 %s</div>'
            '<div class="wmeta">Asset %s · Priority %s</div></div>'
            % (_esc(wo.get("wonum")), _esc((wo.get("description") or "")[:60]),
               _esc(wo.get("asset_id")), wo.get("priority")))
    for d in reversed(dispatches[-4:]):
        sev = d.get("severity", "MED")
        html_parts.append(
            '<div class="write-card" style="border-left-color:#d97706">'
            '<div class="wid">%s · Dispatch</div>'
            '<div class="wtitle">🚛 %s</div>'
            '<div class="wmeta">%s · <span class="badge badge-%s">%s</span></div></div>'
            % (_esc(d.get("dispatch_id")), _esc(d.get("alert_type")),
               _esc(d.get("zone")), sev, sev))
    if not html_parts:
        return '<div style="color:#94a3b8;font-size:0.9rem;padding:8px">No commits yet — agents are watching.</div>'
    return "".join(html_parts)


def render_handoff_dashboard(result):
    stats = result["stats"]
    events = result["events"]
    incidents = result.get("incidents") or []
    ncrs = result.get("quality_ncrs") or []
    flags = result.get("inventory_flags") or []
    wos = result.get("work_orders") or []
    dispatches = result.get("dispatch_alerts") or []

    committed = stats.get("committed", 0)
    held = stats.get("held", 0)
    autonomy = stats.get("autonomy", 100)
    moments = stats.get("moments", 0)

    headline = "Shift completed with %d committed action(s) across %d moments · %.0f%% autonomous." % (
        committed, moments, autonomy)
    if committed == 0 and held == 0:
        headline = "Quiet shift — %d moments reviewed, floor remained clear." % moments

    flow = (
        '<div class="flowchart">'
        '<div class="flow-node done">📹 Watch CCTV</div><span class="flow-arrow">→</span>'
        '<div class="flow-node done">🎯 Supervisor Decides</div><span class="flow-arrow">→</span>'
        '<div class="flow-node done">🚀 Deploy Agent</div><span class="flow-arrow">→</span>'
        '<div class="flow-node done">⚖️ Critic Audits</div><span class="flow-arrow">→</span>'
        '<div class="flow-node highlight">✅ System Write</div>'
        '</div>'
    )

    summary = (
        '<div class="summary-grid">'
        '<div class="sum-card"><div class="num" style="color:#ea580c">%d</div><div class="lbl">Safety</div></div>'
        '<div class="sum-card"><div class="num" style="color:#9333ea">%d</div><div class="lbl">Quality NCRs</div></div>'
        '<div class="sum-card"><div class="num" style="color:#059669">%d</div><div class="lbl">Inventory</div></div>'
        '<div class="sum-card"><div class="num" style="color:#2563eb">%d</div><div class="lbl">Work Orders</div></div>'
        '<div class="sum-card"><div class="num" style="color:#d97706">%d</div><div class="lbl">Dispatch</div></div>'
        '</div>' % (len(incidents), len(ncrs), len(flags), len(wos), len(dispatches))
    )

    rows = []
    for i, ev in enumerate(events, 1):
        route = ev.get("route") or "none"
        summary = ev.get("summary") or "No notable change detected."
        if route == "none" or not ev.get("event_type"):
            status = '<span class="badge badge-clear">Clear</span>'
            agent = ""
        elif ev.get("committed"):
            status = '<span class="badge badge-ok">Committed</span>'
            agent = ROUTE_MAP.get(route, (None, None, route.title(), None))[2]
        elif ev.get("held_reason"):
            status = '<span class="badge badge-hold">Held</span>'
            agent = ROUTE_MAP.get(route, (None, None, route.title(), None))[2]
        else:
            status = '<span class="badge badge-clear">Reviewed</span>'
            agent = ""
        sev = ev.get("committed_severity") or ev.get("severity") or ""
        sev_badge = ('<span class="badge badge-%s">%s</span>' % (sev, sev)
                     if sev in ("LOW", "MED", "HIGH", "CRITICAL") else "")
        agent_html = ('<span class="ma">%s</span>' % _esc(agent)) if agent else ""
        rows.append(
            '<div class="moment-card">'
            '<div class="moment-head">'
            '<span class="mn">Moment %d</span>%s%s%s</div>'
            '<div class="moment-obs">%s</div></div>'
            % (i, sev_badge, agent_html, status, _esc(summary)))

    moments_html = (
        '<div class="moments-list">%s</div>' % "".join(rows)
        if rows else '<div style="color:#94a3b8;font-size:0.85rem">No moments recorded.</div>'
    )

    def record_block(title, icon, items, fmt_fn):
        if not items:
            return ""
        blocks = ['<div class="records-section"><h3>%s %s</h3>' % (icon, title)]
        for rec in items:
            blocks.append(fmt_fn(rec))
        blocks.append('</div>')
        return "".join(blocks)

    records = ""
    records += record_block("Safety Incidents", "🦺", incidents, lambda r: (
        '<div class="record-row"><span class="ricon">🦺</span><div class="rdetail">'
        '<div class="rtitle">%s</div><div class="rsub">%s · %s · OSHA %s</div></div>'
        '<span class="badge badge-%s">%s</span></div>'
        % (_esc(r.get("hazard_type")), _esc(r.get("incident_id")),
           _esc(r.get("location")), _esc(r.get("osha_category", "")[:30]),
           r.get("severity", "MED"), r.get("severity", "MED"))))
    records += record_block("Quality NCRs", "🔬", ncrs, lambda r: (
        '<div class="record-row"><span class="ricon">🔬</span><div class="rdetail">'
        '<div class="rtitle">%s</div><div class="rsub">%s · Batch %s · %s</div></div></div>'
        % (_esc(r.get("defect_type")), _esc(r.get("ncr_id")),
           _esc(r.get("batch_id")), _esc(r.get("disposition")))))
    records += record_block("Inventory Flags", "📦", flags, lambda r: (
        '<div class="record-row"><span class="ricon">📦</span><div class="rdetail">'
        '<div class="rtitle">%s</div><div class="rsub">%s · %s</div></div></div>'
        % (_esc(r.get("issue_type")), _esc(r.get("flag_id")), _esc(r.get("location")))))
    records += record_block("Work Orders", "🔧", wos, lambda r: (
        '<div class="record-row"><span class="ricon">🔧</span><div class="rdetail">'
        '<div class="rtitle">%s</div><div class="rsub">%s · Asset %s · P%s</div></div></div>'
        % (_esc((r.get("description") or "")[:70]), _esc(r.get("wonum")),
           _esc(r.get("asset_id")), r.get("priority"))))
    records += record_block("Dispatch Alerts", "🚛", dispatches, lambda r: (
        '<div class="record-row"><span class="ricon">🚛</span><div class="rdetail">'
        '<div class="rtitle">%s</div><div class="rsub">%s · %s</div></div></div>'
        % (_esc(r.get("alert_type")), _esc(r.get("dispatch_id")), _esc(r.get("zone")))))

    if not records:
        records = '<div style="color:#94a3b8;padding:12px">No system records committed this shift.</div>'

    return (
        '<div class="handoff-hero"><h2>📊 End-of-Shift Report</h2><p>%s</p></div>'
        '%s%s%s%s'
        % (headline, flow, summary, moments_html, records)
    )


def build_sequences(uploaded_file, use_sample_footage, interval):
    if uploaded_file is not None and not use_sample_footage:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp.close()
        out_dir = tempfile.mkdtemp(prefix="zapdos_frames_")
        frames = videolib.extract_frames(tmp.name, every_n_seconds=interval,
                                         out_dir=out_dir, seq_name="cam")
        seqs = []
        for i in range(0, len(frames), MAX_PER_SEQ):
            seqs.append(("moment %d" % (len(seqs) + 1), frames[i:i + MAX_PER_SEQ]))
            if len(seqs) >= MAX_SEQS:
                break
        return seqs, tmp.name
    return discover_sequences(), None


def now_ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.markdown(render_header(), unsafe_allow_html=True)

uc1, uc2, uc3, uc4 = st.columns([3, 1, 1, 1.2])
with uc1:
    uploaded = st.file_uploader(
        "📹 Drop your factory CCTV clip here",
        type=["mp4", "mov", "avi", "m4v"],
        help="Warehouse, production line, or dock camera footage works best.",
    )
with uc2:
    every_n = st.selectbox("Frame interval", [1, 2, 3, 5], index=1,
                           format_func=lambda s: "Every %ds" % s)
with uc3:
    use_sample = st.toggle("Demo frames", value=not bool(uploaded),
                           help="Use bundled sample images in data/")
with uc4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    start = st.button("▶  START SHIFT", use_container_width=True)

kpi_ph = st.empty()
kpi_ph.markdown(render_kpi_row(["0"] * 6 + ["—"]), unsafe_allow_html=True)

col_vid, col_side = st.columns([1.55, 1])

with col_vid:
    cam_video = st.empty()
    strip_ph = st.empty()

with col_side:
    side_ph = st.empty()
    side_ph.markdown(
        render_side_panel(AlertFeed().render(), render_writes([], [], [], [], [])),
        unsafe_allow_html=True)

crew_ph = st.empty()
crew_ph.markdown(
    render_agent_crew({a["key"]: "idle" for a in AGENTS}, DEFAULT_MSG),
    unsafe_allow_html=True)

handoff_ph = st.empty()

if not start:
    st.caption("Upload a clip or enable **Demo frames**, then press **Start Shift**.")


def update_crew(statuses, messages, timestamps):
    crew_ph.markdown(render_agent_crew(statuses, messages, timestamps), unsafe_allow_html=True)


if start:
    try:
        sequences, video_path = build_sequences(uploaded, use_sample, every_n)
    except videolib.VideoError as exc:
        st.error(str(exc))
        st.stop()

    if not sequences:
        st.error("No frames found. Upload a video or add seqN_NN.jpg to data/.")
        st.stop()

    if video_path:
        cam_video.video(video_path, autoplay=True, loop=True, muted=True)
    else:
        cam_video.markdown(
            '<div class="video-placeholder">Demo mode — analyzing frames below. '
            'Upload a video for live playback.</div>',
            unsafe_allow_html=True)

    statuses = {a["key"]: "idle" for a in AGENTS}
    messages = dict(DEFAULT_MSG)
    timestamps = {}
    alerts = AlertFeed()
    kpi_vals = ["0", "0", "0", "0", "0", "0", "—"]
    incidents, ncrs, flags, wos, dispatches = [], [], [], [], []
    state = {"thumbs": [], "current_frames": [], "moment": 0}

    def set_msg(key, text):
        messages[key] = text
        timestamps[key] = now_ts()

    def refresh_side():
        side_ph.markdown(
            render_side_panel(alerts.render(), render_writes(
                incidents, ncrs, flags, wos, dispatches)),
            unsafe_allow_html=True)

    def alert(severity, text, moment=None):
        alerts.add(severity, text, moment=moment or state.get("moment"))
        refresh_side()

    def _moment(ev):
        return ev.get("index") or state.get("moment")

    def _short(s, n=72):
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    def thumb_from_path(path, route=None, label="", use_cv=False):
        if use_cv and detlib:
            try:
                return base64.b64encode(detlib.annotate_frame(path)).decode()
            except Exception:
                pass
        if route:
            color = ROUTE_BGR.get(route, (99, 102, 241))
            buf = videolib.annotate_frame(path, label or "ANALYZING", color=color, sublabel="")
            return base64.b64encode(buf).decode()
        return _thumb_b64(path)

    def emit(ev):
        k = ev["kind"]
        m = _moment(ev)

        if k == "moment_start":
            state["moment"] = ev["index"]
            state["current_frames"] = ev["frames"]
            state["thumbs"] = [_thumb_b64(p) for p in ev["frames"]]
            for i, p in enumerate(ev["frames"]):
                state["thumbs"][i] = thumb_from_path(p, use_cv=True)
                strip_ph.markdown(render_strip(state["thumbs"], active_idx=i), unsafe_allow_html=True)
                kpi_vals[0] = str(int(kpi_vals[0]) + 1)
                kpi_ph.markdown(render_kpi_row(kpi_vals), unsafe_allow_html=True)
                time.sleep(0.12)
            strip_ph.markdown(render_strip(state["thumbs"], done=True), unsafe_allow_html=True)

        elif k == "cv_scan":
            scan = ev.get("scan", {})
            if scan.get("critical"):
                for vtype in scan.get("violation_types", []):
                    sev = "CRITICAL" if "Fall" in vtype else "HIGH"
                    alert(sev, "PPE: %s" % vtype, moment=m)

        elif k == "supervisor_thinking":
            statuses.update({a["key"]: "idle" for a in AGENTS})
            statuses["supervisor"] = "thinking"
            set_msg("supervisor", "Reviewing frames…")
            update_crew(statuses, messages, timestamps)

        elif k == "supervisor_observe":
            obs = ev["obs"]
            statuses["supervisor"] = "active"
            sev = (obs.get("severity") or "MED").upper()
            if obs.get("event_detected"):
                set_msg("supervisor", _short(obs.get("summary") or obs.get("reasoning"), 200))
                alert(sev, _short(obs.get("summary") or obs.get("event_type"), 80), moment=m)
            else:
                set_msg("supervisor", "All clear this moment.")

        elif k == "nothing":
            statuses.update({a["key"]: "idle" for a in AGENTS})
            set_msg("supervisor", "All clear.")
            for ag in AGENTS:
                if ag["key"] != "supervisor":
                    messages[ag["key"]] = ag["default"]
            update_crew(statuses, messages, timestamps)

        elif k == "deploy":
            route = ev.get("route", "safety")
            statuses.update({a["key"]: "idle" for a in AGENTS})
            statuses["supervisor"] = "active"
            statuses[route] = "thinking"
            set_msg("supervisor", "Deploying %s." % ev["role"])
            set_msg(route, "Investigating…")
            update_crew(statuses, messages, timestamps)
            st.toast("Deploying %s" % ev["role"], icon="🚀")

        elif k == "subagent_propose":
            sys = ev["system"]
            statuses[sys] = "active"
            v = ev["action"]["vlm"]
            head = (v.get("hazard") or v.get("defect") or v.get("issue")
                    or v.get("fault") or v.get("alert_type") or "Issue")
            sev = (v.get("severity") or "MED").upper()
            set_msg(sys, _short(head + ". " + (v.get("reasoning") or ""), 200))
            update_crew(statuses, messages, timestamps)
            if state.get("current_frames"):
                label = ROUTE_MAP.get(sys, (None, None, sys.title(), None))[2]
                p = state["current_frames"][-1]
                state["thumbs"][-1] = thumb_from_path(p, sys, label)
                strip_ph.markdown(render_strip(state["thumbs"], done=True), unsafe_allow_html=True)
            alert(sev, "%s: %s" % (ev["role"], _short(head, 60)), moment=m)

        elif k == "critic_verdict":
            v = ev["verdict"]
            if not v.get("approved"):
                alert("HELD", _short(v.get("reason"), 80), moment=m)
                set_msg("supervisor", "Held — %s" % _short(v.get("reason"), 120))
            update_crew(statuses, messages, timestamps)

        elif k == "commit":
            rec = ev["record"]
            sys = ev["system"]
            sev = (rec.get("severity") or "MED").upper()
            if sys == "maintenance":
                pri = rec.get("priority", 3)
                sev = {1: "CRITICAL", 2: "HIGH", 3: "MED", 4: "LOW"}.get(pri, "MED")
            if sys == "safety":
                incidents.append(rec)
                kpi_vals[1] = str(len(incidents))
            elif sys == "quality":
                ncrs.append(rec)
                kpi_vals[2] = str(len(ncrs))
            elif sys == "inventory":
                flags.append(rec)
                kpi_vals[3] = str(len(flags))
            elif sys == "maintenance":
                wos.append(rec)
                kpi_vals[4] = str(len(wos))
            elif sys == "dispatch":
                dispatches.append(rec)
                kpi_vals[5] = str(len(dispatches))
            kpi_ph.markdown(render_kpi_row(kpi_vals), unsafe_allow_html=True)
            head = (rec.get("hazard_type") or rec.get("defect_type") or rec.get("issue_type")
                    or rec.get("description") or rec.get("alert_type") or "logged")
            alert(sev, "✅ %s — %s" % (SYSTEM_LABEL.get(sys, sys), _short(head, 50)), moment=m)
            set_msg(sys, "Committed to %s." % SYSTEM_LABEL.get(sys, sys))
            statuses.update({a["key"]: "idle" for a in AGENTS})
            statuses[sys] = "active"
            update_crew(statuses, messages, timestamps)
            st.toast("Written to %s" % SYSTEM_LABEL.get(sys, sys), icon="✅")
            refresh_side()

        elif k == "hold":
            set_msg("supervisor", "Held — %s" % _short(ev.get("reason"), 120))
            statuses.update({a["key"]: "idle" for a in AGENTS})
            update_crew(statuses, messages, timestamps)
            alert("HELD", _short(ev.get("reason"), 80), moment=m)

        elif k == "handoff_thinking":
            statuses["supervisor"] = "thinking"
            set_msg("supervisor", "Building shift report…")
            update_crew(statuses, messages, timestamps)

    try:
        result = run_shift(sequences, emit=emit)
    except VLMError as exc:
        st.error("VLM endpoint error — no fabricated outputs.\n\n%s" % exc)
        st.stop()

    kpi_vals[6] = "%.0f%%" % result["stats"]["autonomy"]
    w = result["stats"].get("writes", {})
    kpi_vals[1] = str(w.get("incidents", len(incidents)))
    kpi_vals[2] = str(w.get("quality_ncrs", len(ncrs)))
    kpi_vals[3] = str(w.get("inventory_flags", len(flags)))
    kpi_vals[4] = str(w.get("work_orders", len(wos)))
    kpi_vals[5] = str(w.get("dispatch_alerts", len(dispatches)))
    kpi_ph.markdown(render_kpi_row(kpi_vals), unsafe_allow_html=True)

    # Sync writes panel with final session results
    incidents = result.get("incidents") or incidents
    ncrs = result.get("quality_ncrs") or ncrs
    flags = result.get("inventory_flags") or flags
    wos = result.get("work_orders") or wos
    dispatches = result.get("dispatch_alerts") or dispatches
    refresh_side()

    for ag in AGENTS:
        messages[ag["key"]] = "Shift complete. Standing by for next crew."
        statuses[ag["key"]] = "idle"
    update_crew(statuses, messages, timestamps)

    handoff_ph.markdown(render_handoff_dashboard(result), unsafe_allow_html=True)
    with st.expander("📝 Full narrative handoff (exportable)"):
        st.markdown(result["handoff"])
        st.download_button("Download handoff (.md)", result["handoff"],
                           file_name="shift_handoff.md", mime="text/markdown")
