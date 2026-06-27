# ⚡ ZAPDOS LABS — Autonomous Floor Command

Six industrial agents orchestrated by a frontier **Qwen3-VL** model over live CCTV.
Watch → Decide → Act → Report. Every decision is a live VLM call — no hardcoded vision rules.

## The crew (6 floor roles)

| # | Agent | System | Does |
|---|-------|--------|------|
| 01 | 🦺 **Safety Officer** | SafetyCulture | PPE, hazards, OSHA incidents |
| 02 | 🔬 **Quality Inspector** | MasterControl | Defects, quarantine, NCRs |
| 03 | 🎯 **Shift Supervisor** | — | Orchestrator: watches feed, deploys specialists |
| 04 | 📦 **Inventory Clerk** | Manhattan WMS | Damage, miscounts, FEFO flags |
| 05 | 🔧 **Maintenance Tech** | Maximo | Equipment faults, work orders |
| 06 | 🚛 **Floor Dispatcher** | TMS | Forklift/pedestrian conflicts, dock flow |

Plus **⚖️ Critic** — independent VLM audit before any system write.

## Run it locally

```bash
pip install -r requirements.txt
# VLM credentials are in the committed `env` file (auto-loaded)

python3 -m streamlit run app.py   # dashboard (upload CCTV clip)
python3 run.py                    # terminal demo on data/ frames
```

## Deploy (GitHub + live app)

**GitHub Pages (`*.github.io`) only hosts static HTML** — it cannot run this Streamlit app, Python, YOLO, or VLM calls. Use two pieces:

| What | Where | URL |
|------|--------|-----|
| Landing page | GitHub Pages (`docs/`) | `https://haashirazimi.github.io/temp_antler/` |
| Live dashboard | [Streamlit Community Cloud](https://share.streamlit.io) | `https://<your-app>.streamlit.app` |

### 1. Enable GitHub Pages

Repo → **Settings** → **Pages** → Source: **GitHub Actions** (workflow in `.github/workflows/pages.yml` deploys `docs/` on every push to `main`).

### 2. Deploy the dashboard on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → repo `HaashirAzimi/temp_antler`, branch `main`, main file `app.py`.
3. Deploy — credentials load from the committed `env` file. Update the button link in `docs/index.html` to your Streamlit URL.

**Dashboard:** upload a factory clip (or use sample frames in `data/`), hit **▶ START SHIFT**.
Video plays while agents scan frames, the hub lights up, and incidents/NCRs/WOs land on the board.

## Demo pitch (30 sec)

1. *"We didn't train a model — we built a brain that runs the floor."*
2. *"Six specialists, one supervisor, one critic — all reasoning live on your CCTV."*
3. *"Watch it deploy Safety for a missing vest, Quality for a line drift, Dispatch for a near-miss."*

## Files

| File | Role |
|------|------|
| `app.py` | Zapdos-themed control room UI |
| `agents.py` | 6 agents + critic + `run_shift()` |
| `vlm.py` | OpenAI-compatible VLM client |
| `video.py` | Frame extraction + live overlay |
| `schemas.py` | Factory system record builders |
| `data/` | Sample frames (`seqN_NN.jpg`) |
| `output/` | Committed records + handoff |
