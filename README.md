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
(`seqN_NN.jpg`) |
| `output/` | Committed records + handoff |
