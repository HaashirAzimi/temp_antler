"""
schemas.py — payload builders that mirror real factory systems.

These produce dicts shaped like the records IBM Maximo (work orders),
SafetyCulture (incidents), MasterControl (quality), Manhattan WMS
(inventory), and a TMS dispatch board actually store.
"""

import uuid
from datetime import datetime, timezone


def _short_id():
    return uuid.uuid4().hex[:8].upper()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm_severity(severity):
    sev = str(severity).upper().strip()
    aliases = {"MEDIUM": "MED", "MODERATE": "MED", "CRIT": "CRITICAL",
               "SEVERE": "CRITICAL"}
    sev = aliases.get(sev, sev)
    if sev not in ("LOW", "MED", "HIGH", "CRITICAL"):
        sev = "MED"
    return sev


def make_maximo_work_order(asset_id, fault, priority, reported_by):
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
    sev = _norm_severity(severity)
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


def make_mastercontrol_reject(defect, severity, batch_id, disposition):
    sev = _norm_severity(severity)
    return {
        "ncr_id": "NCR-" + _short_id(),
        "defect_type": defect,
        "severity": sev,
        "batch_id": batch_id or "BATCH-UNKNOWN",
        "disposition": disposition or "QUARANTINE",
        "reported_at": _now_iso(),
        "status": "OPEN",
        "capa_required": sev in ("HIGH", "CRITICAL"),
    }


def make_wms_inventory_flag(issue, sku, location, variance):
    return {
        "flag_id": "INV-" + _short_id(),
        "issue_type": issue,
        "sku": sku or "SKU-UNKNOWN",
        "location": location or "Unknown",
        "variance_units": variance,
        "reported_at": _now_iso(),
        "status": "OPEN",
        "fefo_risk": "near_expiry" in (issue or "").lower(),
    }


def make_dispatch_alert(alert_type, zone, vehicle, action):
    return {
        "dispatch_id": "DSP-" + _short_id(),
        "alert_type": alert_type,
        "zone": zone or "Unknown",
        "vehicle_id": vehicle or "N/A",
        "recommended_action": action or "Hold traffic",
        "reported_at": _now_iso(),
        "status": "ACTIVE",
    }
