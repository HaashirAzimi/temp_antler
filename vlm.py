"""
vlm.py — thin client for an OpenAI-compatible Qwen3-VL endpoint.

All reasoning in this project happens here: we send text + (optionally) a
sequence of images to the vision-language model and let IT do the thinking.
There are no pixel-level rules anywhere in this codebase.
"""

import os
import base64
import json
import shutil
import tempfile
import subprocess
import mimetypes
import requests

# Load VLLM_URL / VLLM_KEY / VLLM_MODEL from a .env file so the project works in
# any shell (terminal demo or Streamlit) without manual `source`. Optional dep.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class VLMError(RuntimeError):
    """Raised when the VLM endpoint is unreachable or returns garbage."""


# This endpoint's context window is only 8192 tokens. A vision model turns
# pixels into tokens, so a few full-res frames blow the budget (and the server
# silently crushes whatever does fit into unreadable thumbnails). We downscale
# every frame to a controlled size so each costs a small, predictable number of
# tokens — that lets the whole frame SEQUENCE fit while staying sharp enough for
# the model to actually see hazards. Override with VLM_MAX_IMAGE_DIM if needed.
MAX_IMAGE_DIM = int(os.environ.get("VLM_MAX_IMAGE_DIM", "1024"))
_HAVE_SIPS = shutil.which("sips") is not None


# --- config from environment -------------------------------------------------

def _config():
    url = os.environ.get("VLLM_URL", "").rstrip("/")
    key = os.environ.get("VLLM_KEY", "")
    model = os.environ.get("VLLM_MODEL", "")
    missing = [name for name, val in
               (("VLLM_URL", url), ("VLLM_KEY", key), ("VLLM_MODEL", model))
               if not val]
    if missing:
        raise VLMError(
            "Missing required environment variable(s): " + ", ".join(missing) +
            ".\nSet them (see .env) before running. Example:\n"
            "  export VLLM_URL=http://host:8080\n"
            "  export VLLM_KEY=your-token\n"
            "  export VLLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
        )
    return url, key, model


# --- image helpers -----------------------------------------------------------

def _resized_copy(path, tmpdir):
    """
    Return a path to a downscaled JPEG copy whose longest side is MAX_IMAGE_DIM,
    using macOS `sips` (no extra Python deps). Falls back to the original if
    sips is unavailable or fails.
    """
    if not _HAVE_SIPS:
        return path
    out = os.path.join(tmpdir, "r_" + os.path.basename(path) + ".jpg")
    try:
        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-Z", str(MAX_IMAGE_DIM),
             path, "-o", out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30,
        )
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
    except (subprocess.SubprocessError, OSError):
        pass
    return path


def _image_to_data_url(path):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return "data:%s;base64,%s" % (mime, b64)


def _build_messages(prompt, image_paths, json_mode, tmpdir, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    content = []
    if image_paths:
        for p in image_paths:
            small = _resized_copy(p, tmpdir)
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(small)},
            })
    text = prompt
    if json_mode:
        text += (
            "\n\nRespond with ONLY valid JSON. No markdown, no code fences, "
            "no commentary before or after. Just the JSON object."
        )
    content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})
    return messages


# --- low level POST ----------------------------------------------------------

def _post(url, key, model, messages):
    endpoint = url + "/v1/chat/completions"
    headers = {
        # The endpoint returns HTTP 400 "Unsupported Media Type" without this.
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    resp = requests.post(endpoint, headers=headers,
                         data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        raise VLMError(
            "VLM endpoint returned HTTP %s: %s"
            % (resp.status_code, resp.text[:500])
        )
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise VLMError("Unexpected response shape from VLM: %s"
                       % json.dumps(data)[:500]) from exc


def _strip_fences(text):
    """Tolerate the model wrapping JSON in ```json ... ``` fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        if t.startswith("json"):
            t = t[4:]
    return t.strip()


# --- public API --------------------------------------------------------------

def ask_vlm(prompt, image_paths=None, json_mode=False, system=None):
    """
    Ask the vision-language model a question.

    prompt       : the instruction / question text
    image_paths  : optional list of image file paths, sent in order
    json_mode    : if True, instruct the model to return ONLY JSON and parse it,
                   returning a Python dict. Retries once on parse failure.
    system       : optional system-prompt persona (gives each agent its voice)

    Returns the model's text (str) or a parsed dict when json_mode=True.
    Raises VLMError on network/endpoint failure after one retry.
    """
    url, key, model = _config()
    tmpdir = tempfile.mkdtemp(prefix="vlm_frames_")
    try:
        messages = _build_messages(prompt, image_paths, json_mode, tmpdir,
                                   system)

        # network call, retry once on failure
        last_err = None
        raw = None
        for attempt in range(2):
            try:
                raw = _post(url, key, model, messages)
                break
            except (requests.RequestException, VLMError) as exc:
                last_err = exc
        if raw is None:
            raise VLMError(
                "VLM call failed after retry. Last error: %s" % last_err
            )

        if not json_mode:
            return raw

        # parse JSON, with one self-correct retry
        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            fix_messages = _build_messages(
                "Your previous response was not valid JSON. Here it is:\n\n"
                + raw +
                "\n\nReturn the corrected, valid JSON object only.",
                None, True, tmpdir, system,
            )
            try:
                fixed = _post(url, key, model, fix_messages)
                return json.loads(_strip_fences(fixed))
            except (requests.RequestException, VLMError,
                    json.JSONDecodeError) as exc:
                raise VLMError(
                    "VLM did not return valid JSON even after a fix attempt. "
                    "Last raw output:\n%s" % raw
                ) from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
