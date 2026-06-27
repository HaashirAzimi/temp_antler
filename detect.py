"""
detect.py — trained PPE / safety YOLO detector (Roboflow-origin weights via HF).

Runs before VLM calls to ground the supervisor in real bounding-box detections:
hard hats, vests, NO-Hardhat, NO-Safety Vest, falls, etc.
"""

import os
from functools import lru_cache

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

_MODEL = None
_MODEL_REPO = os.environ.get(
    "PPE_MODEL_REPO", "Hexmon/vyra-yolo-ppe-detection")
_MODEL_FILE = os.environ.get("PPE_MODEL_FILE", "best.pt")
_CONF = float(os.environ.get("PPE_CONF", "0.38"))

# Classes that trigger safety routing
SAFETY_VIOLATIONS = {
    "NO-Hardhat", "NO-Hard Hat", "no-hardhat", "no_hard_hat",
    "NO-Safety Vest", "NO-Mask", "NO-Goggles", "NO-Gloves",
    "Fall-Detected", "fall-detected",
}
SAFETY_POSITIVE = {"Hardhat", "Safety Vest", "Goggles", "Mask", "Gloves"}


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
        _MODEL = YOLO(path)
        return _MODEL
    except Exception as exc:
        raise RuntimeError(
            "PPE detector failed to load. Install: pip install ultralytics huggingface_hub\n"
            "Error: %s" % exc
        ) from exc


def scan_frames(frame_paths, conf=None):
    """
    Run PPE YOLO on each frame. Returns aggregated findings for the sequence.
    """
    conf = conf if conf is not None else _CONF
    if not frame_paths:
        return _empty_result()

    try:
        model = _load_model()
    except RuntimeError as exc:
        return _empty_result(error=str(exc))

    all_dets = []
    violations = []
    positives = []

    for path in frame_paths:
        try:
            results = model.predict(path, conf=conf, verbose=False)
        except Exception:
            continue
        if not results:
            continue
        r0 = results[0]
        if r0.boxes is None:
            continue
        for box in r0.boxes:
            cls_id = int(box.cls)
            name = model.names.get(cls_id, str(cls_id))
            score = float(box.conf)
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            det = {"class": name, "confidence": round(score, 3), "box": xyxy,
                   "frame": os.path.basename(path)}
            all_dets.append(det)
            if name in SAFETY_VIOLATIONS or name.upper().startswith("NO-"):
                violations.append(det)
            elif name in SAFETY_POSITIVE:
                positives.append(det)

    # de-dupe violation types for summary
    vio_types = {}
    for v in violations:
        vio_types[v["class"]] = max(vio_types.get(v["class"], 0), v["confidence"])

    critical = bool(violations)
    if violations:
        parts = ["%s (%.0f%% conf)" % (k, v * 100) for k, v in vio_types.items()]
        summary = "TRAINED PPE MODEL detected violations: " + ", ".join(parts)
    elif positives:
        pos_types = sorted(set(p["class"] for p in positives))
        summary = "PPE model: compliant gear visible (%s)" % ", ".join(pos_types)
    elif all_dets:
        summary = "PPE model: " + ", ".join(
            sorted(set(d["class"] for d in all_dets)))
    else:
        summary = "PPE model: no persons/PPE detected in these frames."

    return {
        "detections": all_dets,
        "violations": violations,
        "positives": positives,
        "critical": critical,
        "summary": summary,
        "violation_types": list(vio_types.keys()),
    }


def annotate_frame(path, conf=None):
    """Return JPEG bytes with YOLO boxes drawn (for timeline thumbnails)."""
    if not HAVE_CV2:
        with open(path, "rb") as fh:
            return fh.read()
    try:
        model = _load_model()
        conf = conf if conf is not None else _CONF
        results = model.predict(path, conf=conf, verbose=False)
        if results and results[0].plot is not None:
            plotted = results[0].plot()
            ok, buf = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return buf.tobytes()
    except Exception:
        pass
    with open(path, "rb") as fh:
        return fh.read()


def format_for_vlm(scan_result):
    """Text block injected into supervisor / safety VLM prompts."""
    lines = [scan_result.get("summary", "")]
    for v in scan_result.get("violations", [])[:8]:
        lines.append("  - %s @ %s (%.0f%%)" % (
            v["class"], v["frame"], v["confidence"] * 100))
    if scan_result.get("error"):
        lines.append("(Detector unavailable: %s)" % scan_result["error"])
    return "\n".join(lines)


def _empty_result(error=None):
    return {
        "detections": [], "violations": [], "positives": [],
        "critical": False,
        "summary": "PPE detector not run." if not error else error,
        "violation_types": [], "error": error,
    }
