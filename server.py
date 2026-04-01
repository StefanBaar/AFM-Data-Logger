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
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
import json as _json
import threading
import multiprocessing
import os

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

# ── Multiprocessing worker (must be at module level to be picklable) ───────────

def _stage_worker(folder_str: str) -> tuple:
    """Run scan_tdms_stage in a subprocess. Returns (folder_str, result_dict)."""
    try:
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent))
        from afm_io import scan_tdms_stage as _scan
        result = _scan(_Path(folder_str))
        return (folder_str, result)
    except Exception as e:
        return (folder_str, {})


# ── Local import ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from afm_io import (discover_all, update_dataset_meta,
                     _find_measurement_configs, _find_fv_configs,
                     parse_pf_path, parse_fv_path,
                     parse_pf_config, parse_fv_config,
                     load_comments, scan_tdms_stage,
                     _dataset_is_filled, _create_nan_config)

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
    """Non-streaming fallback — returns all datasets at once."""
    if not _current_root:
        return _safe_json({"error": "no_root", "datasets": []})
    root = Path(_current_root)
    if not root.exists():
        return _safe_json({"error": f"Path does not exist: {_current_root}", "datasets": []})
    try:
        datasets = discover_all(root)
        return _safe_json({"error": None, "datasets": datasets})
    except Exception as e:
        import traceback
        return _safe_json({"error": str(e), "traceback": traceback.format_exc(), "datasets": []})


@app.get("/api/scan-stream")
def scan_stream():
    """Two-phase SSE stream.

    Phase 1 — fast config scan (sequential):
      Reads config.txt for every folder and emits datasets immediately.
      U_dist/U_max/U_min are None at this point.
      Ends with: {type:'ready', datasets:[...]}

    Phase 2 — background TDMS stage scan (multiprocessing, N/2 cores):
      Only PF folders with a real ForceCurve.tdms and no valid cache are processed.
      Each result emits: {type:'stage', folder:str, u_dist:f, u_max:f, u_min:f}
      Ends with: {type:'stage_done'}

    Error event: {type:'error', message:str}
    Progress:    {type:'progress', done:N, total:N, phase:1|2, label:str}
    """
    if not _current_root:
        def err():
            yield 'data: ' + _json.dumps({"type":"error","message":"no_root"}) + '\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    root = Path(_current_root)
    if not root.exists():
        def err2():
            yield 'data: ' + _json.dumps({"type":"error","message":f"Path not found: {_current_root}"}) + '\n\n'
        return StreamingResponse(err2(), media_type="text/event-stream")

    def ev(obj):
        body = _json.dumps(_sanitize(obj), ensure_ascii=False, default=str)
        return f'data: {body}\n\n'

    def generate():
        try:
            # ── Phase 1: fast config scan ─────────────────────────────────────
            pf_cfgs = list(_find_measurement_configs(root / "PF", "ForceCurve.tdms"))
            fv_cfgs = list(_find_fv_configs(root / "FV"))
            total1  = len(pf_cfgs) + len(fv_cfgs)
            done1   = 0
            datasets = []

            yield ev({"type":"progress","done":0,"total":total1,"phase":1,
                      "label":f"Found {total1} folders — reading configs…"})

            # PF config scan (no TDMS yet)
            for cfg in pf_cfgs:
                try:
                    path_info = parse_pf_path(cfg)
                    cfg_data  = parse_pf_config(cfg)
                    comments  = load_comments(cfg.parent)
                    datasets.append({
                        "mode":"PF","config_path":str(cfg),"folder":str(cfg.parent),
                        "date":path_info["date"],"time":path_info["time"],
                        "meas_id":path_info["meas_id"],
                        "cantilever":comments.get("cantilever") or cfg_data.get("cantilever"),
                        "frequency_hz":cfg_data["frequency_hz"],"n_samples":cfg_data["n_samples"],
                        "x_length":cfg_data["x_length"],"y_length":cfg_data["y_length"],
                        "x_step":cfg_data["x_step"],"y_step":cfg_data["y_step"],
                        "u_amplitude":cfg_data["u_amplitude"],"u_trigger":cfg_data["u_trigger"],
                        "phase_start":cfg_data["phase_start"],"phase_end":cfg_data["phase_end"],
                        "u_dist":None,"u_max":None,"u_min":None,
                        "comments":comments.get("comments",""),
                        "has_config":_dataset_is_filled(cfg_data,comments),
                    })
                except Exception:
                    pass
                done1 += 1
                yield ev({"type":"progress","done":done1,"total":total1,"phase":1,
                          "label":cfg.parent.name})

            # FV config scan
            for cfg in fv_cfgs:
                try:
                    path_info = parse_fv_path(cfg)
                    cfg_data  = parse_fv_config(cfg)
                    comments  = load_comments(cfg.parent)
                    datasets.append({
                        "mode":"FV","config_path":str(cfg),"folder":str(cfg.parent),
                        "date":path_info["date"],"time":path_info["time"],
                        "sample_name":comments.get("sample_name") or path_info["sample_name"],
                        "sample_name_original":path_info["sample_name"],
                        "cantilever":comments.get("cantilever") or cfg_data.get("cantilever"),
                        "u_trigger":cfg_data["u_trigger"],"app_speed":cfg_data["app_speed"],
                        "ret_speed":cfg_data["ret_speed"],"x_length":cfg_data["x_length"],
                        "y_length":cfg_data["y_length"],"x_step":cfg_data["x_step"],
                        "y_step":cfg_data["y_step"],"num_approach":cfg_data["num_approach"],
                        "num_retract":cfg_data["num_retract"],"loop_time":cfg_data["loop_time"],
                        "ret_length":cfg_data["ret_length"],"filter":cfg_data["filter"],
                        "velocity_app":cfg_data["velocity_app"],"velocity_ret":cfg_data["velocity_ret"],
                        "comments":comments.get("comments",""),
                        "has_config":_dataset_is_filled(cfg_data,comments),
                    })
                except Exception:
                    pass
                done1 += 1
                yield ev({"type":"progress","done":done1,"total":total1,"phase":1,
                          "label":cfg.parent.name})

            # Sort and send datasets immediately — UI renders now
            datasets.sort(
                key=lambda d: (d.get("date") or "0000-00-00", d.get("time") or "00:00:00"),
                reverse=True
            )
            yield ev({"type":"ready","datasets":datasets})

            # ── Phase 2: TDMS stage scan via multiprocessing ──────────────────
            # Only PF folders with a TDMS file; skip if cache already valid
            pf_folders_to_scan = []
            for d in datasets:
                if d["mode"] != "PF":
                    continue
                folder = Path(d["folder"])
                tdms   = folder / "ForceCurve.tdms"
                cache  = folder / "stage_cache.txt"
                if not tdms.exists():
                    continue
                # Skip if cache is fresh
                if cache.exists() and cache.stat().st_mtime >= tdms.stat().st_mtime:
                    try:
                        cached = _json.loads(cache.read_text(encoding="utf-8"))
                        if "u_dist" in cached:
                            # Emit cached result immediately
                            yield ev({"type":"stage","folder":str(folder),
                                      "u_dist":cached["u_dist"],
                                      "u_max":cached["u_max"],
                                      "u_min":cached["u_min"]})
                            continue
                    except Exception:
                        pass
                pf_folders_to_scan.append(str(folder))

            total2 = len(pf_folders_to_scan)
            if total2 == 0:
                yield ev({"type":"stage_done"})
                return

            n_cores = max(1, os.cpu_count() // 2)
            yield ev({"type":"progress","done":0,"total":total2,"phase":2,
                      "label":f"Reading TDMS stage data ({n_cores} cores)…"})

            done2 = 0
            # Use imap_unordered so results stream back as they finish
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_cores) as pool:
                for folder_str, result in pool.imap_unordered(
                        _stage_worker, pf_folders_to_scan, chunksize=1):
                    done2 += 1
                    if result:
                        yield ev({"type":"stage","folder":folder_str,
                                  "u_dist":result.get("u_dist"),
                                  "u_max":result.get("u_max"),
                                  "u_min":result.get("u_min")})
                    yield ev({"type":"progress","done":done2,"total":total2,"phase":2,
                              "label":Path(folder_str).name})

            yield ev({"type":"stage_done"})

        except Exception as e:
            import traceback
            yield ev({"type":"error","message":str(e),"traceback":traceback.format_exc()})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


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


@app.get("/api/maps")
def get_maps(folder: str):
    """Build spatial maps from sensor files and TDMS force curves.

    All data points are irregularly spaced — positions come from Xsensors.txt /
    Ysensors.txt.  Everything is interpolated onto a regular grid of at most
    100x100 using nearest-neighbour griddata.

    Maps returned:
      ZSamplePID   — from ZSamplePID.txt   (one value per measurement point)
      ZTipoffsets  — from ZTipoffsets.txt
      D_max        — max deflection per force curve  (from TDMS ch0)
      D_min        — min deflection per force curve
      Zs_range     — stage voltage range per FC      (from TDMS ch2)

    Also returns x_coords / y_coords arrays (grid axes) for the crosshair.
    """
    import math as _math
    import numpy as _np

    meas = Path(folder)
    if not meas.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    result = {"folder": folder, "maps": {}, "n_curves": None,
              "x_coords": None, "y_coords": None}

    # ── Read spatial positions ────────────────────────────────────────────────
    x_path = meas / "Xsensors.txt"
    y_path = meas / "Ysensors.txt"
    if not x_path.exists() or not y_path.exists():
        return _safe_json(result)

    try:
        xs = _np.loadtxt(str(x_path)).ravel()
        ys = _np.loadtxt(str(y_path)).ravel()
    except Exception:
        return _safe_json(result)

    n_pts = len(xs)
    if n_pts == 0 or len(ys) != n_pts:
        return _safe_json(result)

    # ── Build output grid ─────────────────────────────────────────────────────
    GRID = min(100, max(10, int(_np.sqrt(n_pts))))
    xi = _np.linspace(xs.min(), xs.max(), GRID)
    yi = _np.linspace(ys.min(), ys.max(), GRID)
    Xi, Yi = _np.meshgrid(xi, yi)
    pts = (xs, ys)

    def gridmap(values):
        """Grid irregular values onto GRID×GRID, return sanitised flat list."""
        from scipy.interpolate import griddata as _gd
        g = _gd(pts, values, (Xi, Yi), method="nearest")
        flat = g.ravel()
        clean = [None if (_math.isnan(v) or _math.isinf(v)) else round(float(v), 5)
                 for v in flat]
        vvalid = [v for v in clean if v is not None]
        vmin = min(vvalid) if vvalid else 0.0
        vmax = max(vvalid) if vvalid else 1.0
        return {"data": clean, "rows": GRID, "cols": GRID,
                "n": len(clean), "vmin": vmin, "vmax": vmax}

    # ── Sensor text files ─────────────────────────────────────────────────────
    for fname in ["ZSamplePID.txt", "ZTipoffsets.txt"]:
        p = meas / fname
        if not p.exists():
            continue
        try:
            vals = _np.loadtxt(str(p)).ravel()
            if len(vals) == n_pts:
                result["maps"][fname] = gridmap(vals)
        except Exception:
            pass

    # ── TDMS force curve maps ─────────────────────────────────────────────────
    tdms_path = meas / "ForceCurve.tdms"
    if tdms_path.exists():
        try:
            from nptdms import TdmsFile as _TF
            tdms  = _TF.open(str(tdms_path))
            chs   = tdms["Forcecurve"].channels()
            n_samp = len(chs[0][:]) // n_pts   # samples per FC

            if n_samp > 0:
                result["n_curves"] = n_pts

                # Read full channels once, reshape to (n_pts, n_samp)
                D  = _np.array(chs[0][:n_pts * n_samp]).reshape(n_pts, n_samp)
                Zs = _np.array(chs[2][:n_pts * n_samp]).reshape(n_pts, n_samp)

                result["maps"]["D_max"]   = gridmap(D.max(axis=1))
                result["maps"]["D_min"]   = gridmap(D.min(axis=1))
                result["maps"]["Zs_range"]= gridmap(Zs.max(axis=1) - Zs.min(axis=1))
        except Exception as e:
            pass   # maps still returned without TDMS-derived ones

    # ── Grid axis coords (for crosshair position) ─────────────────────────────
    result["x_coords"] = [round(float(v), 6) for v in xi.tolist()]
    result["y_coords"] = [round(float(v), 6) for v in yi.tolist()]
    result["x_raw"]    = [round(float(v), 6) for v in xs.tolist()]
    result["y_raw"]    = [round(float(v), 6) for v in ys.tolist()]
    result["grid_n"]   = GRID

    return _safe_json(result)


@app.get("/api/fc")
def get_fc(folder: str, index: int = 0):
    """Return one force curve by measurement-point index.
    Channels: 0=Deflection, 1=ZTip_input, 2=ZSensor.
    Also returns the raw X/Y sensor position for this point (crosshair).
    """
    import math as _math
    meas = Path(folder)
    tdms_path = meas / "ForceCurve.tdms"
    if not tdms_path.exists():
        raise HTTPException(status_code=404, detail="ForceCurve.tdms not found")

    try:
        from nptdms import TdmsFile as _TF
        import numpy as _np

        # n_samples from config
        n_samples = 500
        cfg_path = meas / "config.txt"
        if cfg_path.exists():
            import re as _re
            text = cfg_path.read_text(encoding="utf-8", errors="replace")
            m = _re.search(r"FCあたりのデータ取得点数:\s*([0-9]+)", text)
            if m: n_samples = int(m.group(1))

        tdms = _TF.open(str(tdms_path))
        chs  = tdms["Forcecurve"].channels()
        total = len(chs[0][:])
        n_curves = max(1, total // n_samples)
        idx = max(0, min(index, n_curves - 1))
        s, e = idx * n_samples, (idx + 1) * n_samples

        def clean(arr):
            return [None if (_math.isnan(v) or _math.isinf(v)) else round(float(v), 6)
                    for v in _np.array(arr)]

        d  = clean(chs[0][s:e])   # Deflection
        z  = clean(chs[1][s:e])   # ZTip_input
        zs = clean(chs[2][s:e])   # ZSensor

        # Raw sensor position for this FC index
        x_pos, y_pos = None, None
        xf = meas / "Xsensors.txt"
        yf = meas / "Ysensors.txt"
        if xf.exists() and yf.exists():
            try:
                xs = _np.loadtxt(str(xf)).ravel()
                ys = _np.loadtxt(str(yf)).ravel()
                if idx < len(xs):
                    x_pos = round(float(xs[idx]), 6)
                    y_pos = round(float(ys[idx]), 6)
            except Exception:
                pass

        return _safe_json({"index": idx, "z": z, "d": d, "zs": zs,
                           "n": len(d), "n_curves": n_curves,
                           "x_pos": x_pos, "y_pos": y_pos})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return _safe_json({"status": "ok", "root": _current_root})
