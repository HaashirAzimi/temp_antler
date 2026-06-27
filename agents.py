"""
agents.py — the 6-agent floor crew + critic for a factory shift.

    Shift Supervisor (03)  -> orchestrator: watches feed, deploys specialists
    Safety Officer   (01)  -> EHS: hazards, OSHA, SafetyCulture
    Quality Inspector(02)  -> QC: defects, quarantine, MasterControl NCR
    Inventory Clerk  (04)  -> WMS: miscounts, damage, near-expiry flags
    Maintenance Tech (05)  -> CMMS: equipment faults, Maximo work orders
    Floor Dispatcher (06)  -> TMS: vehicle/pedestrian conflicts, dock flow
    Critic                 -> independent gate before any system write

Every decision is a live VLM call. Python only routes conclusions to systems.
"""

import os
import json

from vlm import ask_vlm
from detect import format_for_vlm, scan_frames
from schemas import (
    make_dispatch_alert,
    make_mastercontrol_reject,
    make_maximo_work_order,
    make_safetyculture_incident,
    make_wms_inventory_flag,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

SEV_ORDER = {"LOW": 1, "MED": 2, "HIGH": 3, "CRITICAL": 4}

EVIDENCE_RULES = (
    "GROUND TRUTH RULES:\n"
    "  - Combine TRAINED PPE DETECTOR output (bounding boxes) with your visual review.\n"
    "  - If the detector reports NO-Hardhat / NO-Safety Vest / Fall-Detected, treat as "
    "real unless frames clearly contradict it — deploy safety.\n"
    "  - Do NOT invent smoke, fire, or leaks without clear pixel evidence.\n"
    "  - Normal forklift operation alone is NOT an incident.\n"
    "  - If no detector hits AND nothing looks wrong, set event_detected=false."
)

# deploy route -> (class, system key, display name, icon, commit filename)
ROUTE_MAP = {
    "safety": ("SafetyOfficer", "safety", "Safety Officer", "🦺", "incidents.json"),
    "quality": ("QualityInspector", "quality", "Quality Inspector", "🔬", "quality_ncrs.json"),
    "inventory": ("InventoryClerk", "inventory", "Inventory Clerk", "📦", "inventory_flags.json"),
    "maintenance": ("MaintenanceTech", "maintenance", "Maintenance Technician", "🔧", "work_orders.json"),
    "dispatch": ("FloorDispatcher", "dispatch", "Floor Dispatcher", "🚛", "dispatch_alerts.json"),
}


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
    if emit:
        emit(dict(kind=kind, **payload))


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
# Shift Supervisor — orchestrator (Role 03)
# =============================================================================

class ShiftSupervisor:
    PERSONA = (
        "You are the SHIFT SUPERVISOR running a factory control room. You watch "
        "the camera feed over time and reason about what is CHANGING across "
        "frames. You do not fix anything yourself — you DEPLOY floor specialists. "
        "You are calm, decisive, and conservative — you only deploy when evidence "
        "is clear. You justify every deployment in one or two sentences a floor "
        "manager would respect."
    )

    def observe(self, frame_paths, cv_scan=None):
        cv_block = ""
        if cv_scan:
            cv_block = (
                "\n\nTRAINED PPE DETECTOR (YOLO, Roboflow-trained weights):\n"
                + format_for_vlm(cv_scan) + "\n"
            )
        prompt = (
            "%s%s\n\n"
            "These images are consecutive frames from one fixed factory camera, "
            "in chronological order (earliest first). Reason about what CHANGES "
            "over time.\n\n"
            "Inspect like a real floor supervisor:\n"
            "  - PEOPLE & PPE: missing hard hat / hi-vis vest — trust detector NO-* hits\n"
            "  - VEHICLES: forklift genuinely endangering a pedestrian (not normal ops)\n"
            "  - QUALITY: visible defects, damaged packaging on line\n"
            "  - INVENTORY: damaged cartons, blocked aisles, overstacking\n"
            "  - EQUIPMENT: ONLY if smoke/fire/leak/gauge-red is clearly visible\n"
            "  - LOGISTICS: real congestion or near-miss, not routine traffic\n\n"
            "deploy=none unless event_detected=true with evidence.\n"
            "  'safety' | 'quality' | 'inventory' | 'maintenance' | 'dispatch' | 'none'\n\n"
            "Return JSON: {\"summary\": str, \"event_detected\": bool, "
            "\"event_type\": snake_case label, "
            "\"severity\": 'LOW'|'MED'|'HIGH'|'CRITICAL', "
            "\"deploy\": one of safety|quality|inventory|maintenance|dispatch|none, "
            "\"reasoning\": specific visible evidence only}"
            % (EVIDENCE_RULES, cv_block)
        )
        obs = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                      system=self.PERSONA)
        obs.setdefault("summary", "")
        obs.setdefault("event_detected", False)
        obs.setdefault("event_type", "unknown")
        obs.setdefault("severity", "MED")
        obs.setdefault("deploy", "none")
        obs.setdefault("reasoning", "")
        if isinstance(obs["deploy"], list):
            obs["deploy"] = obs["deploy"][0] if obs["deploy"] else "none"
        return obs


# =============================================================================
# Role 01 — Safety Officer
# =============================================================================

class SafetyOfficer:
    PERSONA = (
        "You are a certified factory SAFETY OFFICER (EHS). You confirm hazards "
        "from camera frames, classify OSHA category and severity, and protect "
        "people first. You do not cry wolf."
    )

    def handle(self, frame_paths, event):
        cv_block = ""
        cv_scan = event.get("cv_scan")
        if cv_scan:
            cv_block = "\n\nTRAINED PPE DETECTOR:\n" + format_for_vlm(cv_scan) + "\n"
        prompt = (
            "%s%s\n\n"
            "TRAINED PPE DETECTOR output is authoritative for helmet/vest compliance.\n\n"
            "Supervisor deployed you for a possible safety hazard:\n"
            "  event_type: %s\n  note: %s\n\n"
            "Confirm ONLY if clearly visible in frames. If not visible, set "
            "\"hazard\": \"none_confirmed\", \"severity\": \"LOW\", and explain in "
            "reasoning that evidence is insufficient.\n\n%s\n\n"
            "Return JSON: {\"hazard\": str, \"severity\": str, "
            "\"location\": str, \"osha_category\": str, "
            "\"corrective_action\": str, \"slack_message\": str, "
            "\"reasoning\": str, \"clearly_visible\": bool}"
            % (EVIDENCE_RULES, cv_block, event.get("event_type"),
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
            "icon": "🦺",
            "vlm": vlm_out,
            "record": incident,
            "slack_message": vlm_out.get(
                "slack_message", "Safety hazard: %s" % incident["hazard_type"]),
            "slack_channel": "#safety-floor",
        }


# =============================================================================
# Role 02 — Quality Inspector
# =============================================================================

class QualityInspector:
    PERSONA = (
        "You are a QC INSPECTOR on the production floor. You inspect parts and "
        "batches against spec, name defects on rejects, and quarantine bad stock "
        "when a line drifts. You think in first-pass yield and CAPA."
    )

    def handle(self, frame_paths, event):
        prompt = (
            "Supervisor deployed you for a possible quality issue:\n"
            "  event_type: %s\n  note: %s\n\n"
            "Inspect the frames for visible defects, misassembly, damaged "
            "packaging, contamination, or line drift.\n\n"
            "Return JSON: {\"defect\": short name, "
            "\"severity\": 'LOW'|'MED'|'HIGH'|'CRITICAL', "
            "\"batch_id\": best guess e.g. 'LOT-4421', "
            "\"disposition\": 'QUARANTINE'|'REWORK'|'SCRAP'|'HOLD', "
            "\"corrective_action\": immediate step, "
            "\"slack_message\": one-line #quality-line alert, "
            "\"reasoning\": evidence across frames}"
            % (event.get("event_type"),
               event.get("reasoning") or event.get("summary"))
        )
        vlm_out = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)
        ncr = make_mastercontrol_reject(
            defect=vlm_out.get("defect", event.get("event_type")),
            severity=vlm_out.get("severity", event.get("severity")),
            batch_id=vlm_out.get("batch_id"),
            disposition=vlm_out.get("disposition", "QUARANTINE"),
        )
        if vlm_out.get("corrective_action"):
            ncr["corrective_action"] = vlm_out["corrective_action"]
        return {
            "system": "quality",
            "role": "Quality Inspector",
            "icon": "🔬",
            "vlm": vlm_out,
            "record": ncr,
            "slack_message": vlm_out.get(
                "slack_message", "Quality reject: %s" % ncr["defect_type"]),
            "slack_channel": "#quality-line",
        }


# =============================================================================
# Role 04 — Inventory Clerk
# =============================================================================

class InventoryClerk:
    PERSONA = (
        "You are an INVENTORY CLERK keeping system stock aligned with the floor. "
        "You flag damage, miscounts, misplacement, blocked pick faces, and "
        "near-expiry risk. You enforce FEFO and keep records audit-ready."
    )

    def handle(self, frame_paths, event):
        prompt = (
            "Supervisor deployed you for a possible inventory issue:\n"
            "  event_type: %s\n  note: %s\n\n"
            "Look for damaged cartons, wrong placement, overstacking, blocked "
            "aisles, label/batch issues, empty pick faces vs overflow.\n\n"
            "Return JSON: {\"issue\": short issue name, "
            "\"sku\": best guess SKU or pallet ID, "
            "\"location\": aisle/bin e.g. 'Aisle C / Bin 12', "
            "\"variance_units\": integer estimate of units affected (0 if unknown), "
            "\"severity\": 'LOW'|'MED'|'HIGH'|'CRITICAL', "
            "\"corrective_action\": immediate step, "
            "\"slack_message\": one-line #inventory alert, "
            "\"reasoning\": evidence}"
            % (event.get("event_type"),
               event.get("reasoning") or event.get("summary"))
        )
        vlm_out = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)
        flag = make_wms_inventory_flag(
            issue=vlm_out.get("issue", event.get("event_type")),
            sku=vlm_out.get("sku"),
            location=vlm_out.get("location"),
            variance=vlm_out.get("variance_units", 0),
        )
        flag["severity"] = vlm_out.get("severity", event.get("severity", "MED"))
        if vlm_out.get("corrective_action"):
            flag["corrective_action"] = vlm_out["corrective_action"]
        return {
            "system": "inventory",
            "role": "Inventory Clerk",
            "icon": "📦",
            "vlm": vlm_out,
            "record": flag,
            "slack_message": vlm_out.get(
                "slack_message", "Inventory flag: %s" % flag["issue_type"]),
            "slack_channel": "#inventory",
        }


# =============================================================================
# Role 05 — Maintenance Technician
# =============================================================================

class MaintenanceTech:
    PERSONA = (
        "You are a senior MAINTENANCE TECHNICIAN. You diagnose equipment faults "
        "from camera frames and prioritize by operational impact and downtime."
    )

    def handle(self, frame_paths, event):
        prompt = (
            "%s\n\n"
            "Supervisor deployed you for a possible equipment fault:\n"
            "  event_type: %s\n  note: %s\n\n"
            "Read gauges, warning lights, smoke, leaks, abnormal motion.\n"
            "Do NOT report smoke unless you see distinct smoke plumes in the "
            "frames — fog machines, dust, and haze are not equipment faults.\n"
            "If no fault is clearly visible, set \"fault\": \"no_fault_confirmed\" "
            "and \"clearly_visible\": false.\n\n"
            "Priority: 1=line down/safety, 2=urgent, 3=schedule soon, 4=routine.\n\n"
            "Return JSON: {\"asset_id\": tag, \"fault\": str, \"repair\": str, "
            "\"priority\": 1-4, \"slack_message\": str, \"reasoning\": str, "
            "\"clearly_visible\": bool}"
            % (EVIDENCE_RULES, event.get("event_type"),
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
            "icon": "🔧",
            "vlm": vlm_out,
            "record": wo,
            "slack_message": vlm_out.get(
                "slack_message", "Equipment fault: %s" % wo["description"]),
            "slack_channel": "#maintenance",
        }


# =============================================================================
# Role 06 — Floor Dispatcher (custom BYO role)
# =============================================================================

class FloorDispatcher:
    PERSONA = (
        "You are the FLOOR DISPATCHER coordinating vehicles, docks, and pedestrian "
        "flow on a busy plant floor. You prevent near-misses, clear staging "
        "bottlenecks, and keep forklifts on safe routes. You think in zones, "
        "clearance, and throughput."
    )

    def handle(self, frame_paths, event):
        prompt = (
            "Supervisor deployed you for a logistics / vehicle-flow issue:\n"
            "  event_type: %s\n  note: %s\n\n"
            "Assess forklift paths, dock congestion, pedestrian crossings, "
            "staging backups, and near-miss situations across the frames.\n\n"
            "Return JSON: {\"alert_type\": short label e.g. 'near_miss' or "
            "'dock_congestion', "
            "\"zone\": plant zone e.g. 'Dock 3 / Cross-aisle B', "
            "\"vehicle_id\": forklift or truck ID if visible else 'UNKNOWN', "
            "\"recommended_action\": immediate dispatch instruction, "
            "\"severity\": 'LOW'|'MED'|'HIGH'|'CRITICAL', "
            "\"slack_message\": one-line #dispatch alert, "
            "\"reasoning\": evidence}"
            % (event.get("event_type"),
               event.get("reasoning") or event.get("summary"))
        )
        vlm_out = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)
        alert = make_dispatch_alert(
            alert_type=vlm_out.get("alert_type", event.get("event_type")),
            zone=vlm_out.get("zone"),
            vehicle=vlm_out.get("vehicle_id"),
            action=vlm_out.get("recommended_action"),
        )
        alert["severity"] = vlm_out.get("severity", event.get("severity", "MED"))
        return {
            "system": "dispatch",
            "role": "Floor Dispatcher",
            "icon": "🚛",
            "vlm": vlm_out,
            "record": alert,
            "slack_message": vlm_out.get(
                "slack_message", "Dispatch alert: %s" % alert["alert_type"]),
            "slack_channel": "#dispatch-floor",
        }


# =============================================================================
# Critic — autonomy gate
# =============================================================================

class Critic:
    PERSONA = (
        "You are an independent CRITIC auditing automated actions before they "
        "commit to real factory systems. You are skeptical — when in doubt, HOLD. "
        "APPROVE only when the claimed hazard/fault is unmistakably visible in "
        "the frames. Reject hallucinated smoke, fire, leaks, and routine forklift "
        "traffic dressed up as incidents."
    )

    SYSTEM_LABELS = {
        "safety": "SafetyCulture (people-safety incident)",
        "quality": "MasterControl (quality NCR / quarantine)",
        "inventory": "Manhattan WMS (inventory flag)",
        "maintenance": "Maximo (equipment work order)",
        "dispatch": "TMS dispatch board (vehicle/logistics alert)",
    }

    def review(self, proposed_action, frame_paths):
        system = proposed_action["system"]
        target = self.SYSTEM_LABELS.get(system, system)
        prompt = (
            "%s\n\n"
            "Audit this automated action against the camera frames.\n"
            "  1. Is the claimed issue UNMISTAKABLY visible? If not → approved=false\n"
            "  2. Is severity appropriate?\n"
            "  3. Is target system correct? → %s\n\n"
            "Reject smoke/fire/leak/work-order claims unless you can point to "
            "specific visual proof in the frames.\n\n"
            "Proposed:\n%s\n\n"
            "Return JSON: {\"approved\": bool, \"reason\": str, "
            "\"adjusted_severity\": 'LOW'|'MED'|'HIGH'|'CRITICAL' or null}"
            % (EVIDENCE_RULES, target, json.dumps(proposed_action["record"], indent=2))
        )
        verdict = ask_vlm(prompt, image_paths=frame_paths, json_mode=True,
                          system=self.PERSONA)
        verdict.setdefault("approved", False)
        verdict.setdefault("reason", "No reason returned.")
        verdict.setdefault("adjusted_severity", None)
        return verdict


# =============================================================================
# Agent instances
# =============================================================================

_AGENTS = {
    "safety": SafetyOfficer(),
    "quality": QualityInspector(),
    "inventory": InventoryClerk(),
    "maintenance": MaintenanceTech(),
    "dispatch": FloorDispatcher(),
}

_COMMIT_FILES = {k: v[4] for k, v in ROUTE_MAP.items()}


def clear_shift_output():
    """Wipe prior shift artifacts so each run only shows THIS session's writes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in set(_COMMIT_FILES.values()) | {"shift_handoff.md"}:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _commit_action(action):
    fname = _COMMIT_FILES.get(action["system"], "events.json")
    return _append_json(fname, action["record"])


def _apply_adjustment(action, adjusted):
    if not adjusted:
        return
    rec = action["record"]
    if "severity" in rec:
        rec["severity"] = adjusted
    if action["system"] == "safety":
        rec["osha_recordable"] = adjusted in ("HIGH", "CRITICAL")


# =============================================================================
# THE SHIFT LOOP
# =============================================================================

def run_shift(sequences, emit=None, fresh_output=True):
    if fresh_output:
        clear_shift_output()

    supervisor = ShiftSupervisor()
    critic = Critic()

    events = []
    committed = 0
    held = 0
    all_writes = {k: [] for k in _COMMIT_FILES}

    _emit(emit, "shift_start", total=len(sequences),
          sequences=[s for s, _ in sequences])

    for idx, (seq, frames) in enumerate(sequences, 1):
        _emit(emit, "moment_start", seq=seq, index=idx,
              total=len(sequences), frames=frames)

        _emit(emit, "cv_scanning", seq=seq)
        cv_scan = scan_frames(frames)
        _emit(emit, "cv_scan", seq=seq, scan=cv_scan)

        _emit(emit, "supervisor_thinking", seq=seq)
        obs = supervisor.observe(frames, cv_scan=cv_scan)

        # If trained detector found PPE violations, ensure safety review
        if cv_scan.get("critical") and not obs.get("event_detected"):
            obs["event_detected"] = True
            obs["deploy"] = "safety"
            obs["event_type"] = obs.get("event_type") or "ppe_violation"
            obs["severity"] = "HIGH"
            if any("Fall" in v for v in cv_scan.get("violation_types", [])):
                obs["severity"] = "CRITICAL"
            obs["reasoning"] = (
                (obs.get("reasoning") or "") + " | PPE detector: "
                + cv_scan.get("summary", ""))[:500]
            obs["summary"] = cv_scan.get("summary", obs.get("summary", ""))
        obs["cv_scan"] = cv_scan
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

        if route not in _AGENTS:
            _emit(emit, "hold", seq=seq, reason="unknown route '%s'" % route)
            record["held_reason"] = "unknown route"
            held += 1
            events.append(record)
            continue

        meta = ROUTE_MAP[route]
        _emit(emit, "deploy", role=meta[2], icon=meta[3],
              reason=obs.get("reasoning"), seq=seq, route=route)
        action = _AGENTS[route].handle(frames, obs)
        vlm = action.get("vlm") or {}
        if vlm.get("clearly_visible") is False:
            record["held_reason"] = "Specialist could not confirm issue in frames."
            held += 1
            _emit(emit, "hold", seq=seq,
                  reason=record["held_reason"], action=action)
            events.append(record)
            continue
        if vlm.get("hazard") == "none_confirmed" or vlm.get("fault") == "no_fault_confirmed":
            record["held_reason"] = "No confirmed hazard/fault in frames."
            held += 1
            _emit(emit, "hold", seq=seq,
                  reason=record["held_reason"], action=action)
            events.append(record)
            continue

        _emit(emit, "subagent_propose", seq=seq, system=action["system"],
              role=action["role"], icon=action.get("icon", "🤖"), action=action)

        _emit(emit, "critic_thinking", seq=seq)
        verdict = critic.review(action, frames)
        _emit(emit, "critic_verdict", seq=seq, verdict=verdict)

        if verdict.get("adjusted_severity"):
            _apply_adjustment(action, verdict["adjusted_severity"])
            record["severity"] = verdict["adjusted_severity"]

        if verdict.get("approved"):
            _commit_action(action)
            all_writes[route].append(action["record"])
            record["committed"] = True
            record["committed_severity"] = record["severity"]
            record["record"] = action["record"]
            committed += 1
            _emit(emit, "commit", seq=seq, system=action["system"],
                  record=action["record"], slack_message=action["slack_message"],
                  slack_channel=action.get("slack_channel", "#floor"))
        else:
            record["held_reason"] = verdict.get("reason")
            held += 1
            _emit(emit, "hold", seq=seq, reason=verdict.get("reason"),
                  action=action)

        events.append(record)

    session_incidents = all_writes["safety"]
    session_ncrs = all_writes["quality"]
    session_flags = all_writes["inventory"]
    session_work_orders = all_writes["maintenance"]
    session_dispatches = all_writes["dispatch"]

    _emit(emit, "handoff_thinking")
    handoff = generate_handoff(
        events, session_incidents, session_work_orders,
        session_ncrs, session_flags, session_dispatches)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "shift_handoff.md"), "w") as fh:
        fh.write(handoff)

    total_actionable = committed + held
    autonomy = (committed / total_actionable * 100) if total_actionable else 100.0
    stats = {
        "moments": len(sequences), "committed": committed,
        "held": held, "autonomy": autonomy,
        "severity_rollup": _severity_rollup(events),
        "writes": {
            "incidents": len(session_incidents),
            "work_orders": len(session_work_orders),
            "quality_ncrs": len(session_ncrs),
            "inventory_flags": len(session_flags),
            "dispatch_alerts": len(session_dispatches),
        },
    }

    _emit(emit, "shift_end", stats=stats, handoff=handoff,
          incidents=session_incidents, work_orders=session_work_orders,
          quality_ncrs=session_ncrs, inventory_flags=session_flags,
          dispatch_alerts=session_dispatches)

    return {
        "events": events,
        "incidents": session_incidents,
        "work_orders": session_work_orders,
        "quality_ncrs": session_ncrs,
        "inventory_flags": session_flags,
        "dispatch_alerts": session_dispatches,
        "handoff": handoff, "stats": stats,
    }


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


def generate_handoff(events, incidents=None, work_orders=None,
                     quality_ncrs=None, inventory_flags=None,
                     dispatch_alerts=None):
    incidents = incidents if incidents is not None else _load_json("incidents.json")
    work_orders = work_orders if work_orders is not None else _load_json("work_orders.json")
    quality_ncrs = quality_ncrs if quality_ncrs is not None else _load_json("quality_ncrs.json")
    inventory_flags = (inventory_flags if inventory_flags is not None
                     else _load_json("inventory_flags.json"))
    dispatch_alerts = (dispatch_alerts if dispatch_alerts is not None
                       else _load_json("dispatch_alerts.json"))
    context = {
        "shift_events": events,
        "safety_incidents": incidents,
        "quality_ncrs": quality_ncrs,
        "inventory_flags": inventory_flags,
        "maintenance_work_orders": work_orders,
        "dispatch_alerts": dispatch_alerts,
    }
    prompt = (
        "Write the end-of-shift handoff for the supervisor taking over. "
        "Professional Markdown. Headers:\n"
        "  # End-of-Shift Handoff\n"
        "  **Headline:** one sentence\n"
        "  ## Safety Incidents\n"
        "  ## Quality NCRs\n"
        "  ## Inventory Flags\n"
        "  ## Work Orders\n"
        "  ## Dispatch Alerts\n"
        "  ## Watch-Items for Next Shift\n\n"
        "Reference IDs where useful. Do not invent data.\n\n%s"
        % json.dumps(context, indent=2)
    )
    return ask_vlm(prompt, image_paths=None, json_mode=False,
                   system=ShiftSupervisor.PERSONA)
