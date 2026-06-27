#!/usr/bin/env python3
"""
run.py — Factory Shift Supervisor Agent (terminal demo).

A frontier Vision-Language Model orchestrates a 3-agent crew that autonomously
runs a factory shift from a camera feed: the Shift Supervisor watches frames
over time, DEPLOYS a Safety Officer or Maintenance Technician, a Critic signs
off, and approved actions write into Maximo / SafetyCulture. Nothing is trained;
all reasoning is live VLM calls.

Run:  python3 run.py     (no arguments needed)
"""

import os
import re
import sys
import glob
import json
from collections import defaultdict

from vlm import VLMError
from agents import run_shift, OUTPUT_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "data")


# --- pretty printing ---------------------------------------------------------

def banner(text, emoji=""):
    line = "=" * 70
    print("\n" + line)
    print("%s %s" % (emoji, text) if emoji else text)
    print(line)


# --- sequence discovery ------------------------------------------------------

def discover_sequences():
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(FRAMES_DIR, ext))

    groups = defaultdict(list)
    for path in files:
        m = re.match(r"(seq\d+)_", os.path.basename(path), re.IGNORECASE)
        if m:
            groups[m.group(1).lower()].append(path)

    def seq_key(s):
        return int(re.search(r"\d+", s).group())

    def frame_key(p):
        return [int(n) for n in re.findall(r"\d+", os.path.basename(p))]

    return [(seq, sorted(groups[seq], key=frame_key))
            for seq in sorted(groups, key=seq_key)]


# --- console renderer for shift events ---------------------------------------

def console_emit(ev):
    k = ev["kind"]
    if k == "moment_start":
        banner("SHIFT MOMENT %d/%d  —  %s  (%d frames over time)"
               % (ev["index"], ev["total"], ev["seq"], len(ev["frames"])),
               "⏱️")
    elif k == "supervisor_thinking":
        print("\n  🎯 SUPERVISOR analyzing the feed (change over time)...")
    elif k == "supervisor_observe":
        obs = ev["obs"]
        print("     summary  : %s" % obs["summary"])
        print("     reasoning: %s" % obs["reasoning"])
        print("     verdict  : event=%s | type=%s | severity=%s | deploy=%s"
              % (obs["event_detected"], obs["event_type"],
                 obs["severity"], obs["deploy"]))
    elif k == "nothing":
        print("  ✅ Nothing actionable this moment — shift continues.")
    elif k == "deploy":
        print("\n  🚀 SUPERVISOR DEPLOYING %s %s"
              % (ev["icon"], ev["role"]))
        print("     reason: %s" % ev["reason"])
    elif k == "subagent_propose":
        v = ev["action"]["vlm"]
        icons = {"safety": "🦺", "quality": "🔬", "inventory": "📦",
                 "maintenance": "🔧", "dispatch": "🚛"}
        print("\n  %s %s proposes:"
              % (icons.get(ev["system"], "🤖"), ev["role"]))
        print(json.dumps(v, indent=6)[:700])
    elif k == "critic_thinking":
        print("\n  ⚖️  CRITIC reviewing (independent second opinion)...")
    elif k == "critic_verdict":
        v = ev["verdict"]
        print("     approved: %s" % v["approved"])
        print("     reason  : %s" % v["reason"])
        if v.get("adjusted_severity"):
            print("     critic adjusted severity → %s" % v["adjusted_severity"])
    elif k == "commit":
        rec = ev["record"]
        print("\n  ✅ Critic APPROVED — committing to system of record:")
        sys = ev["system"]
        if sys == "safety":
            print("[#safety-floor] 🚨 %s | id=%s"
                  % (ev["slack_message"], rec["incident_id"]))
        elif sys == "quality":
            print("[#quality-line] 🔬 NCR %s | %s"
                  % (rec["ncr_id"], rec["defect_type"]))
        elif sys == "inventory":
            print("[#inventory] 📦 %s | %s"
                  % (rec["flag_id"], rec["issue_type"]))
        elif sys == "dispatch":
            print("[#dispatch-floor] 🚛 %s | %s"
                  % (rec["dispatch_id"], rec["alert_type"]))
        else:
            print("[#maintenance] 🔧 WO %s | asset=%s | %s"
                  % (rec["wonum"], rec["asset_id"], ev["slack_message"]))
    elif k == "hold":
        print("\n  ✋ Action HELD for human review (critic rejected).")
        print("     reason: %s" % ev["reason"])
        print("     → logged, not committed. The human-in-the-loop safety net.")


# --- main --------------------------------------------------------------------

def main():
    banner("ZAPDOS LABS — 6-AGENT FLOOR CREW", "⚡")
    print("  A frontier VLM orchestrates a crew, live, over time.")
    print("  No training, no hardcoded vision rules — reasoning happens now.")
    print("  Model: %s" % os.environ.get("VLLM_MODEL", "(unset)"))

    sequences = discover_sequences()
    if not sequences:
        print("\n❌ No frames found in %s (expected seqN_NN.jpg)." % FRAMES_DIR)
        sys.exit(1)
    print("  Discovered %d shift moment(s): %s"
          % (len(sequences), ", ".join(s for s, _ in sequences)))

    result = run_shift(sequences, emit=console_emit)
    stats = result["stats"]

    banner("SHIFT SUMMARY", "📊")
    print("  Moments reviewed     : %d" % stats["moments"])
    print("  Actions committed    : %d" % stats["committed"])
    print("  Held for human review: %d" % stats["held"])
    print("  Severity rollup      : %s" % stats["severity_rollup"])
    print("  Autonomous decisions : %.0f%% committed without human intervention"
          % stats["autonomy"])

    banner("END-OF-SHIFT HANDOFF REPORT", "📝")
    print(result["handoff"])
    print("\n  💾 Handoff saved to %s"
          % os.path.join(OUTPUT_DIR, "shift_handoff.md"))
    banner("END OF SHIFT", "🏁")


if __name__ == "__main__":
    try:
        main()
    except VLMError as exc:
        print("\n❌ VLM ERROR — the demo cannot continue.\n")
        print(str(exc))
        print("\nThis agent does NOT fabricate model outputs. Fix the endpoint "
              "/ credentials and re-run.")
        sys.exit(2)
