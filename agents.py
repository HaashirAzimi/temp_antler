"""
agents.py — the 3-agent crew that runs a factory shift.

    ShiftSupervisor (orchestrator)  -> watches the feed, decides who to deploy
    SafetyOfficer   (subagent)      -> EHS: hazards, OSHA, SafetyCulture
    MaintenanceTech (subagent)      -> equipment faults, Maximo work orders
    Critic          (autonomy gate) -> independent second opinion before commit

Every decision is a live VLM call with a distinct persona. Nothing is trained,
nothing is a hardcoded rule on pixels. The Python only routes the model's
conclusions to the right "system" and persists them.

The whole shift loop is `run_shift(sequences, emit)`. `emit` is a callback that
receives structured events as they happen, so the SAME loop powers both the
terminal demo (run.py) and the Streamlit control room (app.py).
"""

import os
import json

from vlm import ask_vlm
from schemas import make_maximo_work_order, make_safetyculture_incident

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

SEV_ORDER = {"LOW": 1, "MED": 2, "HIGH": 3, "CRITICAL": 4}


# --- persistence helpers -----------------------------------------------------

def _append_json(filename, record):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    items = []
    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                items = json.load(fh)
            if not isinstance(items, list):
                items = []
        except (json.JSONDecodeError, OSError):
            items = []
    items.append(record)
    with open(path, "w") as fh:
        json.dump(items, fh, indent=2)
    return path


def _emit(emit, kind, **payload):
    """Safely fire an event to the UI/console callback."""
    if emit:
        emit(dict(kind=kind, **payload))


# --- shared OSHA reference (the model CHOOSES from this, we don't decide) -----

OSHA_REFERENCE = """Common OSHA 29 CFR categories to choose the best fit from:
- 1910.132  General PPE requirements
- 1910.135  Head protection (hard hats)
- 1910.133  Eye and face protection
- 1910.138  Hand protection
- 1910.23   Fall protection / floor & wall openings
- 1910.157  Portable fire extinguishers (fire / smoke)
- 1910.176  Materials handling & storage (blocked aisles, unstable stacks)
- 1910.178  Powered industrial trucks (forklifts)
- 1910.147  Lockout/Tagout (energy control)
- 1910.1200 Hazard communication (chemical spill / exposure)"""


# =============================================================================
# 1. Shift Supervisor — the orchestrator (Role 03)
# =============================================================================

class ShiftSupervisor:
    PERSONA = (
        "You are the SHIFT SUPERVISOR running a factory control room. You watch "
        "the camera feed over time and reason about what is CHANGING across "
        "frames. You do not fix anything yourself — you DEPLOY specialists: a "
        "Safety Officer for people-safety hazards, a Maintenance Technician for "
        "equipment faults. You are calm, decisive, and you justify every "
        "deployment in one or two sentences a floor manager would respect."
    )

    def observe(self, frame_paths):
        prompt = (
            "These images are consecutive frames from one fixed factory camera, "
            "in chronological order (earliest first). Reason about what CHANGES "
            "over time — movement, posture, PPE, equipment state, gauges, smoke, "
            "spills, forklifts near people. A real event reveals itself through "
            "change, not a single snapshot.\n\n"
            "Inspect carefully and specifically, like a real floor supervisor:\n"
            "  - Each PERSON: are they wearing required PPE for the area "
            "(hard hat, hi-vis vest, eye protection)? Any unsafe act, working "
            "at height without fall protection, or person in a danger zone?\n"
            "  - Each VEHICLE (forklift/pallet jack): operated safely, forks "
            "lowered when parked, clear of pedestrians, load stable?\n"
            "  - The ENVIRONMENT: smoke, fire, spills, blocked aisles, unstable "
            "stacks, gauges or warning lights, leaks, a stalled line.\n\n"
            "Do not invent hazards that are not there — but do not gloss over a "
            "real one. Decide who to deploy:\n"
            "  'safety'      -> a person-safety hazard (missing PPE, forklift "
            "near a person, fire/smoke, spill, fall risk, person down)\n"
            "  'maintenance' -> an equipment fault (machine smoke, gauge in the "
            "red, leak, stalled/struggling line, fault light)\n"
            "  'none'        -> nothing actionable\n\n"
            "Return JSON with exactly these keys: "
            "{\"summary\": what changes across the frames, "
            "\"event_detected\": bool, "
            "\"event_type\": short snake_case label, "
            "\"severity\": one of 'LOW','MED','HIGH','CRITICAL', "
            "\"deploy\": one of 'safety','maintenance','none', "
            "\"reasoning\": the specific visual evidence for your decision}"
        )
        obs = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                      system=self.PERSONA)
        obs.setdefault("summary", "")
        obs.setdefault("event_detected", False)
        obs.setdefault("event_type", "unknown")
        obs.setdefault("severity", "MED")
        obs.setdefault("deploy", "none")
        obs.setdefault("reasoning", "")
        # tolerate the model returning deploy as a list (brief asked for a list)
        if isinstance(obs["deploy"], list):
            obs["deploy"] = obs["deploy"][0] if obs["deploy"] else "none"
        return obs


# =============================================================================
# 2. Safety Officer — EHS subagent (Role 01)
# =============================================================================

class SafetyOfficer:
    PERSONA = (
        "You are a certified factory SAFETY OFFICER (EHS). You confirm hazards "
        "from camera frames, classify them by OSHA category and severity, and "
        "protect people first. You are precise about OSHA codes and you do not "
        "cry wolf — severity must match what is actually visible."
    )

    def handle(self, frame_paths, event):
        prompt = (
            "The shift supervisor deployed you for a possible safety hazard in "
            "these time-ordered frames:\n  event_type: %s\n  supervisor_note: %s"
            "\n\nConfirm the hazard from the images and classify it.\n\n%s\n\n"
            "Severity guide: LOW=minor/no injury, MED=could cause injury, "
            "HIGH=likely serious injury, CRITICAL=imminent danger to life.\n\n"
            "Return JSON with exactly: "
            "{\"hazard\": short hazard name, "
            "\"severity\": 'LOW'|'MED'|'HIGH'|'CRITICAL', "
            "\"location\": best guess of plant location e.g. 'Aisle B / Line 2', "
            "\"osha_category\": the single best-fit code+name from the list, "
            "\"corrective_action\": the immediate corrective action, "
            "\"slack_message\": one-line #safety-floor alert, "
            "\"reasoning\": why this classification fits the evidence}"
            % (event.get("event_type"),
               event.get("reasoning") or event.get("summary"),
               OSHA_REFERENCE)
        )
        vlm_out = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)

        incident = make_safetyculture_incident(
            hazard=vlm_out.get("hazard", event.get("event_type")),
            severity=vlm_out.get("severity", event.get("severity")),
            location=vlm_out.get("location", "Unknown"),
            osha_category=vlm_out.get("osha_category", "1910.132 General PPE"),
        )
        if vlm_out.get("corrective_action"):
            incident["corrective_action"] = vlm_out["corrective_action"]

        return {
            "system": "safety",
            "role": "Safety Officer",
            "vlm": vlm_out,
            "record": incident,
            "slack_message": vlm_out.get(
                "slack_message", "Safety hazard: %s" % incident["hazard_type"]),
        }


# =============================================================================
# 3. Maintenance Technician — subagent (Role 05)
# =============================================================================

class MaintenanceTech:
    PERSONA = (
        "You are a senior MAINTENANCE TECHNICIAN. You diagnose equipment faults "
        "from camera frames — smoke from a machine, gauges in the red, leaks, "
        "stalled or struggling lines — and you prioritize repairs by real "
        "operational impact. You think in assets, root cause, and downtime."
    )

    def handle(self, frame_paths, event):
        prompt = (
            "The shift supervisor deployed you for a possible equipment fault in "
            "these time-ordered frames:\n  event_type: %s\n  supervisor_note: %s"
            "\n\nRead any gauges, warning lights, smoke, leaks, or abnormal "
            "motion across the frames. Identify the likely asset and propose a "
            "repair.\n\nPriority guide: 1=line down/safety, 2=urgent, "
            "3=schedule soon, 4=routine.\n\n"
            "Return JSON with exactly: "
            "{\"asset_id\": plausible tag e.g. 'PMP-204' or 'MTR-118', "
            "\"fault\": concise fault description, "
            "\"repair\": recommended corrective action, "
            "\"priority\": integer 1-4, "
            "\"slack_message\": one-line #maintenance note, "
            "\"reasoning\": the temporal evidence (what changed across frames)}"
            % (event.get("event_type"),
               event.get("reasoning") or event.get("summary"))
        )
        vlm_out = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)

        wo = make_maximo_work_order(
            asset_id=vlm_out.get("asset_id", "ASSET-UNKNOWN"),
            fault=vlm_out.get("fault", event.get("event_type")),
            priority=vlm_out.get("priority", 3),
            reported_by="shift-supervisor-agent",
        )
        if vlm_out.get("repair"):
            wo["description"] = "%s | Recommended: %s" % (
                wo["description"], vlm_out["repair"])

        return {
            "system": "maintenance",
            "role": "Maintenance Technician",
            "vlm": vlm_out,
            "record": wo,
            "slack_message": vlm_out.get(
                "slack_message", "Equipment fault: %s" % wo["description"]),
        }


# =============================================================================
# 4. Critic — the autonomy gate
# =============================================================================

class Critic:
    PERSONA = (
        "You are an independent QUALITY CRITIC auditing automated actions before "
        "they commit to real factory systems. You are skeptical but fair. You "
        "APPROVE actions whose hazard/fault is clearly visible and reasonably "
        "classified. You HOLD (reject) only when the evidence does not support "
        "the claim, the severity is grossly wrong, or the wrong system was "
        "chosen. A small severity tweak does not require a rejection."
    )

    def review(self, proposed_action, frame_paths):
        record = proposed_action["record"]
        system = proposed_action["system"]
        target = ("SafetyCulture (a people-safety incident)"
                  if system == "safety"
                  else "Maximo (an equipment work order)")
        prompt = (
            "Audit this automated action against the camera frames before it "
            "commits. Rubric:\n"
            "  1. Is the claimed hazard/fault ACTUALLY visible in the frames?\n"
            "  2. Is the severity appropriate (not grossly over/understated)?\n"
            "  3. Is the target system correct? This action targets %s.\n\n"
            "Approve if it is broadly correct (a one-level severity tweak is "
            "fine via adjusted_severity). Hold only if the claim is not "
            "supported, severity is off by 2+ levels, or the wrong system was "
            "chosen.\n\nProposed action:\n%s\n\n"
            "Return JSON with exactly: "
            "{\"approved\": bool, "
            "\"reason\": concise verdict against the rubric, "
            "\"adjusted_severity\": one of 'LOW','MED','HIGH','CRITICAL' if a "
            "minor correction helps, else null}"
            % (target, json.dumps(record, indent=2))
        )
        verdict = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)
        verdict.setdefault("approved", False)
        verdict.setdefault("reason", "No reason returned.")
        verdict.setdefault("adjusted_severity", None)
        return verdict


# =============================================================================
# commit helpers
# =============================================================================

def _commit_safety(action):
    path = _append_json("incidents.json", action["record"])
    return path


def _commit_maintenance(action):
    path = _append_json("work_orders.json", action["record"])
    return path


def _apply_adjustment(action, adjusted):
    """Apply a critic severity tweak to the record before commit."""
    if not adjusted:
        return
    if action["system"] == "safety":
        action["record"]["severity"] = adjusted
        action["record"]["osha_recordable"] = adjusted in ("HIGH", "CRITICAL")


# =============================================================================
# THE SHIFT LOOP — shared by run.py (console) and app.py (UI)
# =============================================================================

def run_shift(sequences, emit=None):
    """
    Run the full autonomous shift over `sequences` (list of (name, [paths])).
    Fires `emit(event_dict)` for every step so a UI/console can render live.
    Returns a results dict with events, incidents, work_orders, handoff, stats.
    """
    supervisor = ShiftSupervisor()
    safety = SafetyOfficer()
    maint = MaintenanceTech()
    critic = Critic()

    events = []
    committed = 0
    held = 0

    _emit(emit, "shift_start", total=len(sequences),
          sequences=[s for s, _ in sequences])

    for idx, (seq, frames) in enumerate(sequences, 1):
        _emit(emit, "moment_start", seq=seq, index=idx,
              total=len(sequences), frames=frames)

        # 1. supervisor temporal observation
        _emit(emit, "supervisor_thinking", seq=seq)
        obs = supervisor.observe(frames)
        _emit(emit, "supervisor_observe", seq=seq, obs=obs)

        record = {
            "sequence": seq, "event_type": obs.get("event_type"),
            "severity": obs.get("severity"), "summary": obs.get("summary"),
            "route": obs.get("deploy"), "committed": False,
        }

        route = (obs.get("deploy") or "none").lower()
        if not obs.get("event_detected") or route == "none":
            _emit(emit, "nothing", seq=seq)
            events.append(record)
            continue

        # 2. supervisor DEPLOYS a subagent (made explicit + visible)
        if route == "safety":
            _emit(emit, "deploy", role="Safety Officer", icon="🦺",
                  reason=obs.get("reasoning"), seq=seq)
            action = safety.handle(frames, obs)
            commit_fn = _commit_safety
        elif route == "maintenance":
            _emit(emit, "deploy", role="Maintenance Technician", icon="🔧",
                  reason=obs.get("reasoning"), seq=seq)
            action = maint.handle(frames, obs)
            commit_fn = _commit_maintenance
        else:
            _emit(emit, "hold", seq=seq, reason="unknown route '%s'" % route)
            record["held_reason"] = "unknown route"
            held += 1
            events.append(record)
            continue

        _emit(emit, "subagent_propose", seq=seq, system=action["system"],
              role=action["role"], action=action)

        # 3. critic independently reviews
        _emit(emit, "critic_thinking", seq=seq)
        verdict = critic.review(action, frames)
        _emit(emit, "critic_verdict", seq=seq, verdict=verdict)

        if verdict.get("adjusted_severity"):
            _apply_adjustment(action, verdict["adjusted_severity"])
            record["severity"] = verdict["adjusted_severity"]

        # 4. commit only if approved
        if verdict.get("approved"):
            commit_fn(action)
            record["committed"] = True
            record["committed_severity"] = record["severity"]
            record["record"] = action["record"]
            committed += 1
            _emit(emit, "commit", seq=seq, system=action["system"],
                  record=action["record"], slack_message=action["slack_message"])
        else:
            record["held_reason"] = verdict.get("reason")
            held += 1
            _emit(emit, "hold", seq=seq, reason=verdict.get("reason"),
                  action=action)

        events.append(record)

    # --- handoff --------------------------------------------------------------
    incidents = _load_json("incidents.json")
    work_orders = _load_json("work_orders.json")
    _emit(emit, "handoff_thinking")
    handoff = generate_handoff(events, incidents, work_orders)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "shift_handoff.md"), "w") as fh:
        fh.write(handoff)

    total_actionable = committed + held
    autonomy = (committed / total_actionable * 100) if total_actionable else 100.0
    stats = {"moments": len(sequences), "committed": committed,
             "held": held, "autonomy": autonomy,
             "severity_rollup": _severity_rollup(events)}

    _emit(emit, "shift_end", stats=stats, handoff=handoff,
          incidents=incidents, work_orders=work_orders)

    return {"events": events, "incidents": incidents,
            "work_orders": work_orders, "handoff": handoff, "stats": stats}


# --- handoff + rollup --------------------------------------------------------

def _load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _severity_rollup(events):
    counts = {}
    for e in events:
        sev = (e.get("committed_severity") or e.get("severity") or "MED").upper()
        counts[sev] = counts.get(sev, 0) + 1
    parts = ["%s:%d" % (k, counts[k])
             for k in sorted(counts, key=lambda s: -SEV_ORDER.get(s, 0))]
    return ", ".join(parts) if parts else "none"


def generate_handoff(events, incidents=None, work_orders=None):
    incidents = incidents if incidents is not None else _load_json("incidents.json")
    work_orders = (work_orders if work_orders is not None
                   else _load_json("work_orders.json"))
    context = {
        "shift_events": events,
        "safety_incidents": incidents,
        "maintenance_work_orders": work_orders,
    }
    prompt = (
        "Write the end-of-shift handoff for the supervisor taking over, using "
        "the structured shift data below. Concise, professional Markdown a real "
        "supervisor would actually hand off. Use these exact section headers:\n"
        "  # End-of-Shift Handoff\n"
        "  **Headline:** one sentence on how the shift went\n"
        "  ## Incidents (count + severity rollup)\n"
        "  ## Work Orders Raised\n"
        "  ## Open Items\n"
        "  ## Watch-Items for Next Shift\n\n"
        "Reference IDs (incident_id / wonum) where useful. Do not invent data "
        "that is not present.\n\nShift data (JSON):\n%s"
        % json.dumps(context, indent=2)
    )
    return ask_vlm(prompt, image_paths=None, json_mode=False,
                   system=ShiftSupervisor.PERSONA)
