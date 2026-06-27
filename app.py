"""
app.py — CORTEX · Autonomous Floor Command Center (Streamlit showpiece).

A live ops dashboard where a frontier VLM orchestrates a 3-agent crew over a
factory camera clip. Upload a video, hit START SHIFT, and watch the Supervisor
reason over time, DEPLOY specialists (the orchestration diagram animates), the
Critic sign off, and real SafetyCulture / Maximo records land on the board.

The agent backend (agents.py / vlm.py / schemas.py / video.py) is untouched.
Every reasoning string shown here comes from a real VLM call. Nothing is faked.

Run:  python3 -m streamlit run app.py
"""

import os
import time
import base64
import tempfile
import datetime

import streamlit as st

try:
    import cv2
except ImportError:
    cv2 = None

import video as videolib
from vlm import VLMError
from agents import run_shift
from run import discover_sequences

MODEL = os.environ.get("VLLM_MODEL", "(set VLLM_MODEL in .env)")
MAX_PER_SEQ = 4      # frames per "shift moment" (keeps each call within context)
MAX_SEQS = 6         # cap total moments so the live demo stays punchy

st.set_page_config(page_title="CORTEX · Floor Command",
                   page_icon="🧠", layout="wide")

# ===========================================================================
# CSS — dark control room, glassmorphism, neon accents
# ===========================================================================
st.markdown("""
<style>
:root{ --bg:#0a0e14; --cyan:#22d3ee; --green:#34d399; --amber:#fbbf24;
       --orange:#fb923c; --red:#f87171; --blue:#60a5fa; --ink:#e6edf3;
       --muted:#7d8da3; }
.stApp{ background:
   radial-gradient(900px 500px at 12% -8%, #0f1b2a 0%, transparent 55%),
   radial-gradient(900px 500px at 100% 0%, #131024 0%, transparent 50%), #0a0e14;
   color:var(--ink); }
#MainMenu, header, footer{ visibility:hidden; }
section.main > div{ padding-top:.4rem; }

.topbar{ display:flex; align-items:center; gap:18px; padding:14px 20px;
  border-radius:16px; margin-bottom:14px;
  background:linear-gradient(90deg, rgba(34,211,238,.10), rgba(52,211,153,.05));
  border:1px solid rgba(255,255,255,.08);
  box-shadow:0 8px 30px rgba(0,0,0,.45); backdrop-filter:blur(10px); }
.brand{ font-size:26px; font-weight:800; letter-spacing:1px;
  background:linear-gradient(90deg,#22d3ee,#34d399);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.brand small{ display:block; font-size:11px; letter-spacing:3px; font-weight:600;
  color:var(--muted); -webkit-text-fill-color:var(--muted); }
.spacer{ flex:1; }
.clock{ font-family:'SF Mono',monospace; color:var(--cyan); font-size:18px; }
.badge-live{ background:rgba(52,211,153,.15); color:var(--green);
  border:1px solid rgba(52,211,153,.4); padding:5px 12px; border-radius:20px;
  font-size:12px; font-weight:700; letter-spacing:1px; }
.badge-live::before{ content:'●'; margin-right:6px; animation:pulse 1.3s infinite;}
@keyframes pulse{ 0%{opacity:1}50%{opacity:.35}100%{opacity:1} }

.panel{ background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07);
  border-radius:16px; padding:14px 16px; backdrop-filter:blur(8px);
  box-shadow:0 8px 24px rgba(0,0,0,.35); }
.panel-h{ font-size:12px; font-weight:700; letter-spacing:2px;
  text-transform:uppercase; color:var(--muted); margin-bottom:10px; }

.strip{ display:flex; gap:8px; overflow-x:auto; padding:6px 2px; }
.strip .fr{ position:relative; border-radius:8px; overflow:hidden;
  border:2px solid rgba(255,255,255,.06); flex:0 0 auto; }
.strip .fr img{ display:block; height:74px; }
.strip .fr.active{ border-color:var(--cyan);
  box-shadow:0 0 14px rgba(34,211,238,.7); }
.strip .fr.active::after{ content:''; position:absolute; left:0; right:0; height:3px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  animation:scan 1.1s linear infinite; }
@keyframes scan{ 0%{top:0}100%{top:100%} }
.strip .fr.done{ border-color:rgba(52,211,153,.5); }

.feed{ height:430px; overflow-y:auto; padding-right:6px;
  font-family:'SF Mono',ui-monospace,monospace; font-size:12.5px; }
.row{ padding:8px 11px; margin-bottom:7px; border-radius:9px;
  background:rgba(255,255,255,.03); border-left:3px solid #30363d;
  animation:fade .35s ease; }
@keyframes fade{ from{opacity:0; transform:translateY(6px)} to{opacity:1} }
.row .t{ color:#56657d; font-size:10px; }
.row b{ color:#fff; }
.row.sup{ border-left-color:var(--cyan); }
.row.deploy{ border-left-color:#a78bfa; background:rgba(167,139,250,.08); }
.row.safety{ border-left-color:var(--orange); }
.row.maint{ border-left-color:var(--blue); }
.row.critic{ border-left-color:var(--amber); }
.row.commit{ border-left-color:var(--green); background:rgba(52,211,153,.08); }
.row.hold{ border-left-color:var(--red); background:rgba(248,113,113,.08); }
.row.ok{ border-left-color:#3a4658; color:var(--muted); }

.card{ border-radius:12px; padding:13px 14px; margin-bottom:11px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-left:5px solid #30363d; animation:fade .4s ease; }
.card .id{ font-family:monospace; font-size:11px; color:var(--muted); }
.card .ttl{ font-weight:700; font-size:14px; margin:4px 0; color:#fff; }
.card .meta{ font-size:11.5px; color:#9aa7ba; line-height:1.5; }
.glow-CRITICAL{ border-left-color:var(--red);
  box-shadow:0 0 16px rgba(248,113,113,.35); }
.glow-HIGH{ border-left-color:var(--orange);
  box-shadow:0 0 14px rgba(251,146,60,.28); }
.glow-MED{ border-left-color:var(--amber); }
.glow-LOW{ border-left-color:var(--blue); }
.pill{ display:inline-block; padding:2px 9px; border-radius:11px; font-size:10.5px;
  font-weight:800; letter-spacing:.5px; }
.pill-CRITICAL{background:var(--red);color:#1a0000}
.pill-HIGH{background:var(--orange);color:#1a0c00}
.pill-MED{background:var(--amber);color:#1a1400}
.pill-LOW{background:var(--blue);color:#001022}

/* orchestration diagram */
.diagram{ display:flex; justify-content:center; }
.node-lbl{ font:600 11px ui-sans-serif,system-ui; }
.edge{ stroke:#243244; stroke-width:2; fill:none; }
.edge.active{ stroke:var(--cyan); stroke-width:3.5; stroke-dasharray:7 5;
  filter:drop-shadow(0 0 7px var(--cyan)); animation:dash .55s linear infinite; }
@keyframes dash{ to{ stroke-dashoffset:-24; } }
.ncirc{ fill:#0e1622; stroke:#2b3a4f; stroke-width:2; transition:.2s; }
.node.active .ncirc{ fill:#10241b; stroke:var(--green); stroke-width:3;
  filter:drop-shadow(0 0 12px var(--green)); }
.node.sup .ncirc{ stroke:#2b4a5f; }
.node.sup.active .ncirc{ stroke:var(--cyan);
  filter:drop-shadow(0 0 14px var(--cyan)); }

.kpi{ text-align:center; padding:10px 6px; border-radius:12px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07); }
.kpi .v{ font-size:24px; font-weight:800; color:#fff; }
.kpi .l{ font-size:10px; letter-spacing:1px; color:var(--muted);
  text-transform:uppercase; }
.handoff{ background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:22px 26px; }
.dropnote{ color:var(--muted); font-size:13px; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# render helpers
# ===========================================================================

def _thumb_b64(path, h=74):
    """Small JPEG thumbnail as base64 for the frame strip."""
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
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


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
        cells.append('<div class="%s"><img src="data:image/jpeg;base64,%s"></div>'
                     % (cls, b))
    return '<div class="strip">%s</div>' % "".join(cells)


NODES = {
    "supervisor": (190, 140, "🎯", "Supervisor", "sup"),
    "safety":     (66, 58, "🦺", "Safety", ""),
    "maintenance": (314, 58, "🔧", "Maint", ""),
    "critic":     (190, 252, "⚖️", "Critic", ""),
}
EDGES = {"safety": (190, 140, 66, 58), "maintenance": (190, 140, 314, 58),
         "critic": (190, 140, 190, 252)}


def render_diagram(active_node=None, active_edge=None):
    parts = ['<div class="diagram"><svg width="380" height="300" '
             'viewBox="0 0 380 300">']
    for key, (x1, y1, x2, y2) in EDGES.items():
        cls = "edge active" if active_edge == key else "edge"
        parts.append('<line class="%s" x1="%d" y1="%d" x2="%d" y2="%d"/>'
                     % (cls, x1, y1, x2, y2))
    for key, (cx, cy, emoji, label, extra) in NODES.items():
        cls = "node %s" % extra
        if active_node == key:
            cls += " active"
        r = 34 if key == "supervisor" else 28
        parts.append('<g class="%s">' % cls)
        parts.append('<circle class="ncirc" cx="%d" cy="%d" r="%d"/>'
                     % (cx, cy, r))
        parts.append('<text x="%d" y="%d" text-anchor="middle" '
                     'font-size="20">%s</text>' % (cx, cy - 2, emoji))
        parts.append('<text class="node-lbl" x="%d" y="%d" text-anchor="middle" '
                     'fill="#cbd5e1">%s</text>' % (cx, cy + 15, label))
        parts.append('</g>')
    parts.append('</svg></div>')
    return "".join(parts)


def render_feed(rows):
    if not rows:
        return ('<div class="feed"><div class="row ok">Awaiting shift start — '
                'upload a clip and press START SHIFT.</div></div>')
    html = ['<div class="feed">']
    for r in rows:
        html.append('<div class="row %s"><span class="t">%s</span> %s</div>'
                    % (r["cls"], r["t"], r["html"]))
    html.append('</div>')
    return "".join(html)


def render_cards(incidents, work_orders):
    if not incidents and not work_orders:
        return '<div class="dropnote">No system writes yet.</div>'
    html = []
    for inc in reversed(incidents):
        sev = inc.get("severity", "MED")
        html.append(
            '<div class="card glow-%s"><span class="id">%s · SafetyCulture</span>'
            '<div class="ttl">🚨 %s</div>'
            '<div class="meta">📍 %s<br>OSHA %s · recordable: %s<br>↳ %s</div>'
            '<div style="margin-top:6px"><span class="pill pill-%s">%s</span></div>'
            '</div>' % (sev, inc.get("incident_id", ""),
                        inc.get("hazard_type", ""), inc.get("location", ""),
                        inc.get("osha_category", ""),
                        inc.get("osha_recordable", False),
                        inc.get("corrective_action", "")[:90], sev, sev))
    for wo in reversed(work_orders):
        pri = wo.get("priority", 3)
        sev = {1: "CRITICAL", 2: "HIGH", 3: "MED", 4: "LOW"}.get(pri, "MED")
        html.append(
            '<div class="card glow-%s"><span class="id">%s · Maximo</span>'
            '<div class="ttl">🔧 %s</div>'
            '<div class="meta">⚙️ asset %s · %s · status %s</div>'
            '<div style="margin-top:6px"><span class="pill pill-%s">PRIORITY %s'
            '</span></div></div>'
            % (sev, wo.get("wonum", ""), wo.get("description", "")[:80],
               wo.get("asset_id", ""), wo.get("worktype", ""),
               wo.get("status", ""), sev, pri))
    return "".join(html)


def kpi(label, value):
    return '<div class="kpi"><div class="v">%s</div><div class="l">%s</div></div>' \
           % (value, label)


# ===========================================================================
# top bar
# ===========================================================================
st.markdown(
    '<div class="topbar">'
    '<div class="brand">🧠 CORTEX<small>AUTONOMOUS FLOOR COMMAND</small></div>'
    '<div class="spacer"></div>'
    '<div class="clock">%s</div>'
    '<div class="badge-live">SHIFT ACTIVE</div>'
    '</div>' % datetime.datetime.now().strftime("%H:%M:%S"),
    unsafe_allow_html=True)

# ===========================================================================
# controls
# ===========================================================================
ctl1, ctl2, ctl3 = st.columns([2.4, 1, 1])
with ctl1:
    uploaded = st.file_uploader(
        "📹 Drop a warehouse / factory camera clip here",
        type=["mp4", "mov", "avi", "m4v"], label_visibility="visible")
with ctl2:
    every_n = st.selectbox("Sample every", [1, 2, 3, 5], index=1,
                           format_func=lambda s: "%ds" % s)
with ctl3:
    st.caption("‎")
    use_sample = st.toggle("Use sample footage", value=not bool(uploaded),
                           help="Run on the bundled data/ frames if you have "
                                "no clip handy.")

start = st.button("▶  START SHIFT", type="primary", use_container_width=True)

# ===========================================================================
# KPI strip
# ===========================================================================
k1, k2, k3, k4, k5 = st.columns(5)
kp_frames = k1.empty(); kp_inc = k2.empty(); kp_wo = k3.empty()
kp_held = k4.empty(); kp_auto = k5.empty()
kp_frames.markdown(kpi("FRAMES ANALYZED", 0), unsafe_allow_html=True)
kp_inc.markdown(kpi("INCIDENTS", 0), unsafe_allow_html=True)
kp_wo.markdown(kpi("WORK ORDERS", 0), unsafe_allow_html=True)
kp_held.markdown(kpi("HELD FOR REVIEW", 0), unsafe_allow_html=True)
kp_auto.markdown(kpi("AUTONOMY", "—"), unsafe_allow_html=True)

st.write("")

# ===========================================================================
# three-column control room
# ===========================================================================
col_cam, col_orch, col_writes = st.columns([1.5, 1.35, 1.0])

with col_cam:
    st.markdown('<div class="panel-h">📹 Floor Camera</div>',
                unsafe_allow_html=True)
    cam_video = st.empty()
    st.markdown('<div class="panel-h" style="margin-top:10px">Frames under '
                'analysis</div>', unsafe_allow_html=True)
    strip_ph = st.empty()
with col_orch:
    st.markdown('<div class="panel-h">🧠 Agent Orchestration</div>',
                unsafe_allow_html=True)
    diagram_ph = st.empty()
    feed_ph = st.empty()
with col_writes:
    st.markdown('<div class="panel-h">📋 Live System Writes</div>',
                unsafe_allow_html=True)
    cards_ph = st.empty()

diagram_ph.markdown(render_diagram(), unsafe_allow_html=True)
feed_ph.markdown(render_feed([]), unsafe_allow_html=True)
cards_ph.markdown(render_cards([], []), unsafe_allow_html=True)
strip_ph.markdown('<div class="dropnote">Frames appear here as the shift '
                  'advances.</div>', unsafe_allow_html=True)

handoff_ph = st.container()


# ===========================================================================
# build sequences (uploaded video OR sample footage)
# ===========================================================================

def build_sequences(uploaded_file, use_sample_footage, interval):
    """Return (sequences, video_path_or_None). sequences = [(name,[paths])]."""
    if uploaded_file is not None and not use_sample_footage:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp.close()
        out_dir = tempfile.mkdtemp(prefix="cortex_frames_")
        frames = videolib.extract_frames(tmp.name, every_n_seconds=interval,
                                         out_dir=out_dir, seq_name="cam")
        seqs = []
        for i in range(0, len(frames), MAX_PER_SEQ):
            seqs.append(("moment %d" % (len(seqs) + 1),
                         frames[i:i + MAX_PER_SEQ]))
            if len(seqs) >= MAX_SEQS:
                break
        return seqs, tmp.name
    # fallback: bundled sample frame sequences
    return discover_sequences(), None


# ===========================================================================
# run the shift
# ===========================================================================
if start:
    try:
        sequences, video_path = build_sequences(uploaded, use_sample, every_n)
    except videolib.VideoError as exc:
        st.error("Could not read that clip: %s" % exc)
        st.stop()

    if not sequences:
        st.error("No frames to analyze. Upload a clip or enable sample footage.")
        st.stop()

    if video_path:
        cam_video.video(video_path)
    else:
        cam_video.markdown('<div class="dropnote">Running on bundled sample '
                           'footage (frame sequences).</div>',
                           unsafe_allow_html=True)

    feed_rows = []
    incidents, work_orders = [], []
    state = {"frames": 0, "committed": 0, "held": 0,
             "thumbs": [], "node": None, "edge": None}

    def push(cls, html, pause=0.0):
        feed_rows.append({"cls": cls, "t": datetime.datetime.now().strftime(
            "%H:%M:%S"), "html": html})
        feed_ph.markdown(render_feed(feed_rows[-40:]), unsafe_allow_html=True)
        if pause:
            time.sleep(pause)

    def set_diagram(node=None, edge=None):
        state["node"], state["edge"] = node, edge
        diagram_ph.markdown(render_diagram(node, edge), unsafe_allow_html=True)

    def emit(ev):
        k = ev["kind"]
        if k == "moment_start":
            state["thumbs"] = [_thumb_b64(p) for p in ev["frames"]]
            push("ok", "<b>━━ Shift moment %d/%d · %s ━━</b>"
                 % (ev["index"], ev["total"], ev["seq"]))
            # animate the supervisor "scanning" across the frames
            for i in range(len(state["thumbs"])):
                strip_ph.markdown(render_strip(state["thumbs"], active_idx=i),
                                  unsafe_allow_html=True)
                state["frames"] += 1
                kp_frames.markdown(kpi("FRAMES ANALYZED", state["frames"]),
                                   unsafe_allow_html=True)
                time.sleep(0.18)
            strip_ph.markdown(render_strip(state["thumbs"], done=True),
                              unsafe_allow_html=True)
        elif k == "supervisor_thinking":
            set_diagram(node="supervisor")
            push("sup", "🎯 <b>SUPERVISOR</b> scanning the floor "
                 "<i>(reasoning over time…)</i>", 0.3)
        elif k == "supervisor_observe":
            obs = ev["obs"]
            push("sup", "🎯 <b>SUPERVISOR</b> — %s<br><span style='color:#9aa7ba'>"
                 "%s</span>" % (obs["summary"], obs["reasoning"]))
        elif k == "nothing":
            set_diagram()
            push("ok", "✅ Floor clear this moment — no action needed.")
        elif k == "deploy":
            edge = "safety" if "Safety" in ev["role"] else "maintenance"
            set_diagram(node=edge, edge=edge)
            st.toast("🚀 Deploying %s" % ev["role"], icon="🚀")
            push("deploy", "🚀 <b>DEPLOYING → %s %s</b><br>"
                 "<span style='color:#c4b5fd'>%s</span>"
                 % (ev["icon"], ev["role"], ev["reason"]), 0.5)
        elif k == "subagent_propose":
            v = ev["action"]["vlm"]
            cls = "safety" if ev["system"] == "safety" else "maint"
            icon = "🦺" if ev["system"] == "safety" else "🔧"
            head = v.get("hazard") or v.get("fault") or ""
            detail = v.get("reasoning") or v.get("repair") or ""
            extra = (" · OSHA %s" % v.get("osha_category")
                     if v.get("osha_category") else
                     (" · priority %s" % v.get("priority")
                      if v.get("priority") else ""))
            push(cls, "%s <b>%s</b> — confirmed: <i>%s</i>%s<br>"
                 "<span style='color:#9aa7ba'>%s</span>"
                 % (icon, ev["role"], head, extra, detail))
        elif k == "critic_thinking":
            set_diagram(node="critic", edge="critic")
            push("critic", "⚖️ <b>CRITIC</b> auditing the proposed action "
                 "<i>(independent review…)</i>", 0.35)
        elif k == "critic_verdict":
            v = ev["verdict"]
            adj = (" · severity → %s" % v["adjusted_severity"]
                   if v.get("adjusted_severity") else "")
            tag = "APPROVED ✅" if v["approved"] else "HELD ✋"
            push("critic", "⚖️ <b>CRITIC: %s</b>%s<br>"
                 "<span style='color:#9aa7ba'>%s</span>"
                 % (tag, adj, v["reason"]))
        elif k == "commit":
            rec = ev["record"]
            if ev["system"] == "safety":
                incidents.append(rec)
                st.toast("🚨 %s incident logged" % rec.get("severity"),
                         icon="🚨")
                push("commit", "✅ <b>COMMITTED → SafetyCulture %s</b><br>"
                     "<span style='color:#9aa7ba'>[#safety-floor] %s</span>"
                     % (rec["incident_id"], ev["slack_message"]))
            else:
                work_orders.append(rec)
                st.toast("🔧 Work order %s created" % rec.get("wonum"),
                         icon="🔧")
                push("commit", "✅ <b>COMMITTED → Maximo %s</b><br>"
                     "<span style='color:#9aa7ba'>[#maintenance] %s</span>"
                     % (rec["wonum"], ev["slack_message"]))
            state["committed"] += 1
            kp_inc.markdown(kpi("INCIDENTS", len(incidents)),
                            unsafe_allow_html=True)
            kp_wo.markdown(kpi("WORK ORDERS", len(work_orders)),
                           unsafe_allow_html=True)
            cards_ph.markdown(render_cards(incidents, work_orders),
                              unsafe_allow_html=True)
            set_diagram()
        elif k == "hold":
            state["held"] += 1
            kp_held.markdown(kpi("HELD FOR REVIEW", state["held"]),
                             unsafe_allow_html=True)
            st.toast("✋ Action held for human review", icon="✋")
            push("hold", "✋ <b>HELD FOR HUMAN REVIEW</b> — not committed<br>"
                 "<span style='color:#9aa7ba'>%s</span>" % ev["reason"])
            set_diagram()
        elif k == "handoff_thinking":
            set_diagram(node="supervisor")
            push("sup", "🎯 <b>SUPERVISOR</b> compiling the end-of-shift "
                 "handoff…", 0.3)

    try:
        result = run_shift(sequences, emit=emit)
    except VLMError as exc:
        st.error("⚠️ VLM ENDPOINT ERROR — Cortex does not fabricate outputs.\n\n"
                 "%s" % exc)
        st.stop()

    set_diagram()
    stats = result["stats"]
    kp_auto.markdown(kpi("AUTONOMY", "%.0f%%" % stats["autonomy"]),
                     unsafe_allow_html=True)

    with handoff_ph:
        st.write("")
        st.markdown('<div class="panel-h">📊 End-of-Shift Handoff</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="handoff">', unsafe_allow_html=True)
        st.markdown(result["handoff"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("📥 Export Handoff (.md)", result["handoff"],
                           file_name="shift_handoff.md", mime="text/markdown")
        st.toast("🏁 Shift complete", icon="🏁")
else:
    st.info("Upload a clip (or toggle **sample footage**), then press "
            "**▶ START SHIFT** to watch the crew run the floor live.")
