"""
server.py — FastAPI backend for AFM Logger
Run with: uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import json as _json

def _sanitize(obj):
    """Clean data for safe JSON serialization:
    - Replace float nan/inf with None (bare NaN/Infinity is invalid JSON)
    - Replace lone surrogate chars from macOS filesystem with ?
    """
    if isinstance(obj, float):
        import math
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, str):
        # Remove lone surrogates that some macOS volumes produce
        return obj.encode("utf-8", "replace").decode("utf-8", "replace")
    if isinstance(obj, dict):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj


def _safe_json(data) -> Response:
    """Serialize to UTF-8 JSON with explicit charset declaration.
    Uses ensure_ascii=False so Japanese paths are real Unicode chars,
    not \\uXXXX escape sequences that confuse Safari round-trips.
    """
    body = _json.dumps(_sanitize(data), ensure_ascii=False, default=str)
    return Response(
        content=body.encode("utf-8"),
        media_type="application/json; charset=utf-8"
    )

# Local import — works from project root
sys.path.insert(0, str(Path(__file__).parent))
from afm_io import discover_all, update_dataset_meta

# ─────────────────────────────────────────────────────────────────────────────
#  App + static files
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AFM Logger", version="1.0")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    body = _json.dumps({"detail": str(exc.detail)}, ensure_ascii=True)
    return Response(content=body, status_code=exc.status_code,
                    media_type="application/json; charset=utf-8")

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    body = _json.dumps({"detail": str(exc)}, ensure_ascii=True)
    return Response(content=body, status_code=500,
                    media_type="application/json; charset=utf-8")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# Default scan root — empty until user sets it
_current_root: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/root")
def get_root():
    return _safe_json({"root": _current_root})


class RootPayload(BaseModel):
    root: str


@app.post("/api/root")
def set_root(payload: RootPayload):
    global _current_root
    p = Path(payload.root)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {payload.root}")
    _current_root = str(p)
    return _safe_json({"root": _current_root})


@app.get("/api/datasets")
def get_datasets():
    if not _current_root:
        return _safe_json({"error": "no_root", "datasets": []})
    root = Path(_current_root)
    if not root.exists():
        return _safe_json({"error": f"Path does not exist: {_current_root}", "datasets": []})
    try:
        datasets = discover_all(root)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return _safe_json({"error": f"Scan error: {str(e)}", "traceback": tb, "datasets": []})
    try:
        return _safe_json({"error": None, "datasets": datasets})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Serialization failed - find which dataset is the problem
        good = []
        bad = []
        for d in datasets:
            try:
                _safe_json({"test": d})
                good.append(d)
            except Exception as de:
                bad.append({"folder": str(d.get("folder","")), "error": str(de)})
        return _safe_json({"error": f"Serialization error: {str(e)}",
                           "traceback": tb,
                           "bad_datasets": bad,
                           "datasets": good})


@app.get("/api/scan-debug")
def scan_debug():
    """Detailed scan diagnostics - open in browser to see exactly what fails."""
    import traceback, math
    if not _current_root:
        return _safe_json({"error": "no_root"})
    root = Path(_current_root)
    results = {"root": _current_root, "steps": []}
    # Step 1: discover
    try:
        datasets = discover_all(root)
        results["steps"].append({"step": "discover_all", "ok": True, "count": len(datasets)})
    except Exception as e:
        results["steps"].append({"step": "discover_all", "ok": False,
                                  "error": str(e), "traceback": traceback.format_exc()})
        return _safe_json(results)
    # Step 2: check each dataset for bad values
    bad = []
    for d in datasets:
        issues = []
        for k, v in d.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                issues.append(f"{k}={v}")
        if issues:
            bad.append({"folder": str(d.get("folder","")), "issues": issues})
    results["steps"].append({"step": "check_values", "ok": len(bad)==0, "bad": bad})
    # Step 3: serialize
    try:
        import json as _j
        body = _j.dumps(_sanitize({"datasets": datasets}), ensure_ascii=True, default=str)
        results["steps"].append({"step": "json_serialize", "ok": True, "bytes": len(body)})
    except Exception as e:
        results["steps"].append({"step": "json_serialize", "ok": False,
                                  "error": str(e), "traceback": traceback.format_exc()})
    return _safe_json(results)


@app.get("/api/debug")
def debug_scan():
    """Returns a diagnostic tree of what the scanner finds under the current root."""
    if not _current_root:
        return _safe_json({"root": None, "error": "no_root"})
    root = Path(_current_root)
    if not root.exists():
        return _safe_json({"root": _current_root, "error": "path_not_found"})

    pf_root = root / "PF"
    fv_root = root / "FV"

    def _names(p):
        try:
            return sorted(c.name for c in p.iterdir())
        except Exception:
            return []

    def scan_dir(p):
        if not p.exists():
            return {"exists": False}
        entries = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name):
                try:
                    children = _names(child) if child.is_dir() else []
                    entries.append({"name": child.name, "is_dir": child.is_dir(), "children": children})
                except Exception:
                    pass
        except Exception as e:
            return {"exists": True, "error": str(e)}
        return {"exists": True, "entries": entries}

    result = {
        "root": _current_root,
        "root_exists": root.exists(),
        "root_contents": _names(root),
        "PF": scan_dir(pf_root),
        "FV": scan_dir(fv_root),
    }
    return _safe_json(result)


class UpdatePayload(BaseModel):
    folder: str
    mode: str
    updates: dict


@app.post("/api/update")
def update_dataset(payload: UpdatePayload):
    result = update_dataset_meta(payload.folder, payload.mode, payload.updates)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return _safe_json(result)


@app.get("/api/health")
def health():
    return _safe_json({"status": "ok", "root": _current_root})
