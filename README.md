# 🧠 CORTEX — Autonomous Floor Command

A frontier **Vision-Language Model** (Qwen3-VL) orchestrates a **3-agent crew**
that autonomously runs a factory shift from a camera feed. Cortex watches the
floor over time, deploys specialists, audits its own decisions, and writes into
real factory systems (Maximo, SafetyCulture).

**We didn't train a model — we built a brain that runs the floor.** Every
decision is a live VLM call happening right now, on these frames. No training,
no hardcoded rules on pixels.

## The crew

| Agent | Role | Does |
|-------|------|------|
| 🎯 **Shift Supervisor** | orchestrator | Watches the feed, reasons about change over time, **deploys** a specialist |
| 🦺 **Safety Officer** | EHS | Confirms hazard, classifies OSHA category + severity, writes a **SafetyCulture** incident |
| 🔧 **Maintenance Tech** | maintenance | Diagnoses equipment fault, sets priority 1–4, writes a **Maximo** work order |
| ⚖️ **Critic** | autonomy gate | Independent 2nd VLM call: approves or **holds for human** before any commit |

## Run it

```bash
pip install -r requirements.txt
# put your keys in .env (VLLM_URL / VLLM_KEY / VLLM_MODEL)

python3 -m streamlit run app.py   # 🧠 CORTEX control room (the centerpiece)
python3 run.py                    # terminal demo of the same shift loop
```

**Using the dashboard:** drop a warehouse/factory clip into the upload zone (or
flip on **Use sample footage**), then hit **▶ START SHIFT**. Watch the camera
feed scan frames, the orchestration diagram light up as the Supervisor deploys
subagents, incident/work-order cards appear in real time, and the end-of-shift
handoff render at the bottom (with a one-click `.md` export).

## 🎤 What to say to the judges

- **"We didn't train anything — we built a brain that runs the floor."** The reasoning you're watching is happening live, right now, on these frames. That survives every follow-up question.
- **"It reasons over time, not single frames."** The supervisor sees the whole sequence and explains what *changed* — that's how it catches a hazard as it develops.
- **"Watch it deploy specialists."** Supervisor spots the event → the orchestration graph lights up → the Safety Officer or Maintenance Tech writes a real incident or work order.
- **"There's a critic, so automation is safe."** A second independent VLM call signs off before anything commits; rejected actions are held for a human — that's the 90–95% autonomy story.

## Files

| File | Role |
|------|------|
| `app.py` | **CORTEX control room** — video upload, animated agent orchestration, live incident/WO cards, handoff (the showpiece) |
| `run.py` | Terminal demo over the same shift loop |
| `agents.py` | The 3-agent crew + critic + shared `run_shift()` loop |
| `vlm.py` | `ask_vlm()` — base64 frames, OpenAI chat POST, personas, JSON mode + retries |
| `video.py` | `extract_frames()` — sample a CCTV clip into `seqN_NN.jpg` frames |
| `schemas.py` | Maximo work order + SafetyCulture incident record builders |
| `data/` | Sample input frames (`seqN_NN.jpg`) |
| `output/` | Generated incidents, work orders, handoff report |
