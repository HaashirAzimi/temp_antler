"""
schemas.py — payload builders that mirror real factory systems.

These produce dicts shaped like the records IBM Maximo (work orders) and
SafetyCulture (incidents) actually store, so the demo writes look authentic
to an operations manager.
"""

import uuid
from datetime import datetime, timezone


def _short_id():
    return uuid.uuid4().hex[:8].upper()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_maximo_work_order(asset_id, fault, priority, reported_by):
    """
    Build an IBM Maximo work order record.

    priority is clamped to Maximo's 1 (highest) .. 4 (lowest) scale.
    status WAPPR = "Waiting on Approval", the default for a new CM order.
    worktype CM = Corrective Maintenance (reactive repair).
    """
    try:
        pri = int(priority)
    except (TypeError, ValueError):
        pri = 3
    pri = max(1, min(4, pri))

    return {
        "wonum": "WO-" + _short_id(),
        "asset_id": asset_id,
        "description": fault,
        "priority": pri,
        "status": "WAPPR",
        "reportedby": reported_by,
        "reportdate": _now_iso(),
        "worktype": "CM",
        "site": "PLANT01",
    }


def make_safetyculture_incident(hazard, severity, location, osha_category):
    """
    Build a SafetyCulture incident record.

    severity is normalized to LOW / MED / HIGH / CRITICAL.
    osha_recordable is inferred from severity (HIGH/CRITICAL are recordable).
    """
    sev = str(severity).upper().strip()
    aliases = {"MEDIUM": "MED", "MODERATE": "MED", "CRIT": "CRITICAL",
               "SEVERE": "CRITICAL"}
    sev = aliases.get(sev, sev)
    if sev not in ("LOW", "MED", "HIGH", "CRITICAL"):
        sev = "MED"

    recordable = sev in ("HIGH", "CRITICAL")

    return {
        "incident_id": "INC-" + _short_id(),
        "title": hazard,
        "hazard_type": hazard,
        "severity": sev,
        "location": location,
        "osha_recordable": recordable,
        "osha_category": osha_category,
        "reported_at": _now_iso(),
        "corrective_action": "Pending assignment",
        "status": "OPEN",
    }
