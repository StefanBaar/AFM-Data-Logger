"""
server.py — FastAPI backend for AFM Logger
Run with: uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
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

def _fv_worker(args) -> tuple:
    """Process one FV LVM file: load, pre_process, get_E.
    args = (lvm_path_str, idx, INVOLS, StIV, k, ran, th, sample, nu, alpha)
    Returns (idx, zm, ds, cp, E)
    """
    try:
        import numpy as _np
        from scipy import signal as _sig
        from pathlib import Path as _P
        lvm_path, idx, INVOLS, StIV, k, ran, th, sample, nu, alpha = args

        # Fast LVM load
        raw = open(lvm_path, 'rb').read().split()
        n = len(raw) // 2
        D = _np.array(raw[:n],   dtype=_np.float32)
        Z = _np.array(raw[n:2*n], dtype=_np.float32)

        # Downsample
        d, z = D[::sample], Z[::sample]
        z = z * StIV
        base_n = max(1, len(d) // ran)

        # Tilt correction + baseline
        try:
            slope = _np.polyfit(z[:base_n], d[:base_n], 1)[0]
        except Exception:
            slope = 0.0
        d = d - slope * z
        d = (d - _np.median(d[:base_n])) * INVOLS
        sig = _np.std(d[:base_n]) * th

        # Find contact point on approach
        dmax = int(_np.argmax(d))
        dm, zm = d[:dmax], z[:dmax]

        if len(dm) >= 53:
            ds = _np.array(_sig.savgol_filter(dm, 51, 11), dtype=_np.float32)
        else:
            ds = dm.copy()

        below = _np.argwhere(ds < sig)
        cp = int(below[-1][0]) if len(below) else max(0, len(zm) - 1)

        # Young's modulus via Sneddon
        E_val = _np.nan
        if cp < len(zm) - 3:
            import math as _math
            Delta_m = (zm[cp:] - zm[cp]) * 1e-6   # um → m
            F_N     = (ds[cp:] - ds[cp]) * k * 1e-6  # um * N/m → N... wait
            # ds is in um (INVOLS converts V→um), k is N/m
            # F [N] = deflection [m] * k [N/m] = ds[um] * 1e-6 * k
            fmax = int(_np.argmax(F_N))
            if fmax >= 3:
                d2 = Delta_m[:fmax] ** 2
                mask = d2 > 0
                if mask.sum() >= 3:
                    try:
                        alpha_r = _math.radians(alpha)
                        slope_s, _ = _np.polyfit(d2[mask], F_N[:fmax][mask], 1)
                        E_val = float(slope_s * _math.pi * (1 - nu**2) /
                                      (2 * _math.tan(alpha_r)))
                        if E_val < 0: E_val = _np.nan
                    except Exception:
                        pass

        zm_f = _np.array(zm, dtype=_np.float32)
        ds_f = _np.array(ds, dtype=_np.float32)
        return (idx, zm_f, ds_f, int(cp), float(E_val))
    except Exception as e:
        return (args[1], _np.array([]), _np.array([]), 0, float('nan'))


def _stage_worker(folder_str: str) -> tuple:
    """Run scan_tdms_stage in a subprocess. Returns (folder_str, result_dict)."""
    try:
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent))
        from afm_io import scan_tdms_stage as _scan
        result = _scan(_Path(folder_str))
        try: _compute_and_cache_maps(_Path(folder_str))
        except Exception: pass
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



# Bump this whenever map computation logic changes — forces all npz to regenerate
_MAP_CACHE_VERSION = 3

_sensor_cache: dict = {}   # folder -> (xs, ys)
_maps_cache:   dict = {}   # folder -> (result_dict, mtime)

# LRU TDMS cache — max 3 folders in RAM at once.
# Each large PF file is 30-90 MB; unlimited caching caused 16 GB+ RAM use.
_MAX_TDMS = 3
_tdms_cache: dict = {}
_tdms_lru:   list = []   # oldest-first access order


def _tdms_put(key, val):
    """Insert into LRU cache, evict oldest if over limit."""
    if key in _tdms_cache:
        _tdms_lru.remove(key)
    _tdms_lru.append(key)
    _tdms_cache[key] = val
    while len(_tdms_lru) > _MAX_TDMS:
        del _tdms_cache[_tdms_lru.pop(0)]   # numpy arrays released → GC frees RAM


def _load_tdms_arrays(meas: Path):
    """Return (D, Z, Zs, n_samp) float32 arrays, LRU-cached (last 3 folders)."""
    import numpy as _np
    tdms_path = meas / "ForceCurve.tdms"
    if not tdms_path.exists():
        return None, None, None, None
    key   = str(meas)
    mtime = tdms_path.stat().st_mtime
    if key in _tdms_cache and _tdms_cache[key][4] == mtime:
        _tdms_lru.remove(key); _tdms_lru.append(key)   # refresh LRU
        D, Z, Zs, n_samp, _ = _tdms_cache[key]
        return D, Z, Zs, n_samp
    try:
        from nptdms import TdmsFile as _TF
        import re as _re, tempfile as _tmp, shutil as _sh
        _mmap = _tmp.mkdtemp(prefix="afm_mm_")
        try:
            tdms = _TF.read(str(tdms_path), memmap_dir=_mmap)
        except Exception:
            tdms = _TF.read(str(tdms_path))
        chs = tdms["Forcecurve"].channels()
        D   = _np.array(chs[0][:], dtype=_np.float32)   # Deflection
        Z   = _np.array(chs[1][:], dtype=_np.float32)   # ZTip_input
        Zs  = _np.array(chs[2][:], dtype=_np.float32) if len(chs)>2 else _np.array([], dtype=_np.float32)
        try: _sh.rmtree(_mmap, ignore_errors=True)
        except Exception: pass
        n_samp = 500
        cfg = meas / "config.txt"
        if cfg.exists():
            m = _re.search(r"FCあたりのデータ取得点数:\s*([0-9]+)",
                           cfg.read_text(encoding="utf-8", errors="replace"))
            if m: n_samp = int(m.group(1))
        _tdms_put(key, (D, Z, Zs, n_samp, mtime))
        return D, Z, Zs, n_samp
    except Exception:
        return None, None, None, None


def _compute_and_cache_maps(meas) -> bool:
    """Pre-compute CP + sensor maps, save as meas/afm_maps.npz.
    Called from Phase 2 worker so maps are instant on first row expand.
    Returns True if cache was written or already valid.
    """
    import numpy as _np
    from scipy.interpolate import griddata as _gd
    from pathlib import Path as _P
    meas = _P(meas)
    npz  = meas / "afm_maps.npz"
    tdms = meas / "ForceCurve.tdms"
    if not tdms.exists(): return False
    if npz.exists() and npz.stat().st_mtime >= tdms.stat().st_mtime:
        return True  # already fresh
    xs, ys = _load_sensors(meas)
    if xs is None: return False
    n_pts  = len(xs)
    D_all, Z_all, _Zs, n_samp = _load_tdms_arrays(meas)
    if D_all is None or n_samp == 0: return False
    GRID   = min(100, max(10, int(_np.sqrt(n_pts))))
    xi = _np.linspace(xs.min(), xs.max(), GRID)
    yi = _np.linspace(ys.min(), ys.max(), GRID)
    Xi, Yi = _np.meshgrid(xi, yi)
    pts = (xs, ys)
    half  = n_samp // 2
    D_app = D_all[:n_pts*n_samp].reshape(n_pts, n_samp)[:, :half]
    Z_app = Z_all[:n_pts*n_samp].reshape(n_pts, n_samp)[:, :half]

    # Return-direction mask (dx < 0)
    dx    = _np.diff(xs, prepend=xs[0])
    rmask = dx < 0
    if rmask.sum() < 4: rmask = _np.ones(n_pts, dtype=bool)
    pts_r = (xs[rmask], ys[rmask])

    # Use only return-direction points (dx < 0) for cleaner maps
    dx   = _np.diff(xs, prepend=xs[0])
    rmask = dx < 0
    pts_r = (xs[rmask], ys[rmask])  # subset positions
    if rmask.sum() < 4:             # too few — fall back to all points
        rmask = _np.ones(n_pts, dtype=bool)
        pts_r = pts
    n_base = max(5, half//5)
    bm = D_app[:, :n_base].mean(axis=1, keepdims=True)
    bs = D_app[:, :n_base].std(axis=1,  keepdims=True).clip(min=1e-6)
    above = D_app > (bm + 5*bs)
    any_a = above.any(axis=1)
    fi    = _np.where(any_a, above.argmax(axis=1), half-1)
    cp    = Z_app[_np.arange(n_pts), fi].copy()
    cp[~any_a]    = Z_app[~any_a].max(axis=1)
    cp[above[:,0]]= Z_app[above[:,0], 0]
    cp_grid = _gd(pts_r, cp[rmask], (Xi, Yi), method="nearest").astype(_np.float32)
    arrays  = {"cp": cp_grid,
               "xi": xi.astype(_np.float32), "yi": yi.astype(_np.float32),
               "x_raw": xs.astype(_np.float32), "y_raw": ys.astype(_np.float32),
               "grid_n": _np.array([GRID], dtype=_np.int32)}
    for fname, key in [("ZSamplePID.txt","zpid"), ("ZTipoffsets.txt","ztip")]:
        p = meas / fname
        if p.exists():
            try:
                v = _fast_load_txt(p).astype(_np.float32)
                if len(v)==n_pts:
                    arrays[key] = _gd(pts_r, v[rmask], (Xi,Yi), method="nearest").astype(_np.float32)
            except Exception: pass
    try:
        arrays["_version"] = _np.array([_MAP_CACHE_VERSION], dtype=_np.int32)
        _np.savez_compressed(str(npz), **arrays)
        return True
    except Exception:
        return False


def _load_sensors(meas):
    """Load Xsensors.txt; always synthesise Y from config.txt.

    Y is synthesised regardless of Ysensors.txt — the Y stage sensor is
    unreliable.  Synthesis:
      nx = XStep (columns), ny = YStep (rows)
      y[i] = row_index × y_range / (ny-1)
      y_range = x_sensor_range × (Y_size_um / X_size_um)
    """
    import numpy as _np, re as _re
    key = str(meas)
    if key in _sensor_cache:
        return _sensor_cache[key]
    xf = meas / "Xsensors.txt"
    if not xf.exists():
        return None, None
    xs = _fast_load_txt(xf)
    n  = len(xs)

    # Always synthesise Y from config
    nx, ny = 1, 1
    x_size_um, y_size_um = 0.0, 0.0
    cfg = meas / "config.txt"
    if cfg.exists():
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
            mx  = _re.search(r"XStep:\s*([0-9]+)", txt)
            my  = _re.search(r"YStep:\s*([0-9]+)", txt)
            mxs = _re.search(r"X.{1,10}?\(.*?m\):\s*([0-9.]+)", txt)
            mys = _re.search(r"Y.{1,10}?\(.*?m\):\s*([0-9.]+)", txt)
            if mx:  nx = max(1, int(mx.group(1)))
            if my:  ny = max(1, int(my.group(1)))
            if mxs: x_size_um = float(mxs.group(1))
            if mys: y_size_um = float(mys.group(1))
        except Exception:
            pass

    # Fallback: guess square grid
    if nx * ny != n:
        nx = max(1, round(_np.sqrt(n)))
        ny = max(1, n // nx)

    x_range = float(xs.max() - xs.min()) if xs.max() != xs.min() else 1.0
    y_range  = x_range * (y_size_um / x_size_um) if x_size_um > 0 and y_size_um > 0 else x_range

    row_idx = _np.arange(n) // nx
    ys = xs.min() + row_idx.astype(_np.float64) * y_range / max(ny - 1, 1)
    ys = ys.astype(_np.float32)

    _sensor_cache[key] = (xs, ys)
    return xs, ys


def _fast_load_txt(path) -> "_np.ndarray":
    """Load a single-column or whitespace-delimited text file 90x faster than np.loadtxt."""
    import numpy as _np
    data = open(str(path), 'rb').read().split()
    return _np.array(data, dtype=float)



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


# ── Persist root path across restarts ────────────────────────────────────────
_CONFIG_FILE = Path(__file__).parent / ".afm_root"

def _load_saved_root() -> str:
    try:
        p = Path(_CONFIG_FILE.read_text(encoding="utf-8").strip())
        return str(p) if p.exists() else ""
    except Exception:
        return ""

def _save_root(root: str):
    try: _CONFIG_FILE.write_text(root, encoding="utf-8")
    except Exception: pass

_current_root: str = _load_saved_root()


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
        # Give a helpful message — drive may not be mounted
        msg = f"Path not found: {payload.root}"
        if "/Volumes/" in payload.root or payload.root.startswith("/mnt/"):
            msg += " — is the drive mounted?"
        elif len(payload.root) >= 2 and payload.root[1] == ":":
            msg += " — is the drive connected?"
        raise HTTPException(status_code=400, detail=msg)
    _current_root = str(p)
    _save_root(_current_root)
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

            n_cores = max(1, round((os.cpu_count() or 1) * 0.8))
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
                              "label":f"{Path(folder_str).name} — U_PID/U_err/U_Smax/U_Smin"})

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

    X positions from Xsensors.txt; Y is always synthesised from config (Y sensor unreliable).  Everything is interpolated onto a regular grid of at most
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

    # Return cached map result if TDMS/sensor files unchanged
    _map_mtime = 0
    for _fn in ["ForceCurve.tdms","Xsensors.txt","ZSamplePID.txt","ZTipoffsets.txt"]:
        _fp = meas / _fn
        if _fp.exists(): _map_mtime = max(_map_mtime, _fp.stat().st_mtime)
    _map_key = str(meas)
    if _map_key in _maps_cache and _maps_cache[_map_key][1] == _map_mtime:
        return _safe_json(_maps_cache[_map_key][0])

    import time as _time, numpy as _np2
    _ts = _time.time()

    result = {"folder": folder, "maps": {}, "n_curves": None,
              "x_coords": None, "y_coords": None, "timings": {}}

    # ── Fast path: pre-computed .npz ─────────────────────────────────────────
    _npz = meas / "afm_maps.npz"
    _tdms2 = meas / "ForceCurve.tdms"
    if _npz.exists() and (not _tdms2.exists() or _npz.stat().st_mtime >= _tdms2.stat().st_mtime):
        try:
            npz = _np2.load(str(_npz))
            if int(npz.get("_version", _np2.array([0]))[0]) != _MAP_CACHE_VERSION:
                raise ValueError("stale cache version")
            GRID2 = int(npz["grid_n"][0])
            def _nm(arr):
                flat=[round(float(v),4) for v in arr.ravel()]
                return {"data":flat,"rows":GRID2,"cols":GRID2,"n":len(flat),
                        "vmin":round(float(arr.min()),4),"vmax":round(float(arr.max()),4)}
            if "cp"   in npz.files: result["maps"]["CP"]             = _nm(npz["cp"])
            if "zpid" in npz.files: result["maps"]["ZSamplePID.txt"] = _nm(npz["zpid"])
            if "ztip" in npz.files: result["maps"]["ZTipoffsets.txt"]= _nm(npz["ztip"])
            result["n_curves"] = len(npz["x_raw"])
            result["x_coords"] = [round(float(v),4) for v in npz["xi"].tolist()]
            result["y_coords"] = [round(float(v),4) for v in npz["yi"].tolist()]
            result["x_raw"]    = [round(float(v),4) for v in npz["x_raw"].tolist()]
            result["y_raw"]    = [round(float(v),4) for v in npz["y_raw"].tolist()]
            result["grid_n"]   = GRID2
            result["timings"]["npz_load"] = round((_time.time()-_ts)*1000)
            _maps_cache[_map_key] = (result, _map_mtime)
            return _safe_json(result)
        except Exception:
            pass

    result["timings"]["npz_miss"] = round((_time.time()-_ts)*1000)
    _ts = _time.time()

    # ── Slow path: compute from raw files ─────────────────────────────────────
    xs, ys = _load_sensors(meas)
    result["timings"]["load_sensors"] = round((_time.time()-_ts)*1000)
    _ts = _time.time()
    if xs is None:
        _maps_cache[_map_key] = (result, _map_mtime)
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

    def gridmap(values, gpts=None, mask=None):
        """Grid irregular values onto GRID×GRID, return sanitised flat list."""
        from scipy.interpolate import griddata as _gd
        _pts = gpts if gpts is not None else pts
        _vals = values[mask] if mask is not None else values
        g = _gd(_pts, _vals, (Xi, Yi), method="nearest")
        flat = g.ravel()
        clean = [None if (_math.isnan(v) or _math.isinf(v)) else round(float(v), 5)
                 for v in flat]
        vvalid = [v for v in clean if v is not None]
        vmin = min(vvalid) if vvalid else 0.0
        vmax = max(vvalid) if vvalid else 1.0
        return {"data": clean, "rows": GRID, "cols": GRID,
                "n": len(clean), "vmin": vmin, "vmax": vmax}

    # ── Return-direction mask (dx < 0 on X sensor) ───────────────────────────
    _dx    = _np.diff(xs, prepend=xs[0])
    rmask  = _dx < 0
    if rmask.sum() < 4: rmask = _np.ones(n_pts, dtype=bool)
    pts_r  = (xs[rmask], ys[rmask])

    # ── Sensor text files ─────────────────────────────────────────────────────
    for fname in ["ZSamplePID.txt", "ZTipoffsets.txt"]:
        p = meas / fname
        if not p.exists():
            continue
        try:
            vals = _fast_load_txt(p)
            if len(vals) == n_pts:
                result["maps"][fname] = gridmap(vals, pts_r, rmask)
        except Exception:
            pass

    result["timings"]["sensor_maps"] = round((_time.time()-_ts)*1000)
    _ts = _time.time()

    # ── TDMS force curve maps ─────────────────────────────────────────────────
    if (meas / "ForceCurve.tdms").exists():
        try:
            D_all, Z_all, _Zs, n_samp = _load_tdms_arrays(meas)
            if D_all is not None and n_samp > 0:
                result["n_curves"] = n_pts
                half  = n_samp // 2
                D_all = D_all[:n_pts * n_samp].reshape(n_pts, n_samp)
                Z_all = Z_all[:n_pts * n_samp].reshape(n_pts, n_samp)
                D_app = D_all[:, :half]
                Z_app = Z_all[:, :half]

                # ── Contact point detection ────────────────────────────────────
                # Baseline = first 20% of approach; CP = first index where
                # D exceeds baseline + 5*sigma.
                # flat curve (no contact): CP = Z_app.max()
                # already in contact:      CP = Z_app[:,0]
                n_base = max(5, half // 5)
                b_mean = D_app[:, :n_base].mean(axis=1, keepdims=True)
                b_std  = D_app[:, :n_base].std(axis=1, keepdims=True).clip(min=1e-6)
                above  = D_app > (b_mean + 5 * b_std)

                any_above  = above.any(axis=1)
                first_idx  = _np.where(any_above, above.argmax(axis=1), half - 1)
                rows       = _np.arange(n_pts)
                cp_z       = Z_app[rows, first_idx]
                cp_z[~any_above]   = Z_app[~any_above].max(axis=1)   # no contact
                cp_z[above[:, 0]]  = Z_app[above[:, 0], 0]           # full contact

                result["maps"]["CP"] = gridmap(cp_z, pts_r, rmask)
        except Exception:
            pass   # CP map skipped on error

    result["timings"]["tdms_cp"] = round((_time.time()-_ts)*1000)

    # ── Grid axis coords (for crosshair position) ─────────────────────────────
    # Use float32 precision (4 sig figs) — sensor positions don't need 6 decimals
    result["x_coords"] = [round(float(v), 4) for v in xi.tolist()]
    result["y_coords"] = [round(float(v), 4) for v in yi.tolist()]
    result["x_raw"]    = [round(float(v), 4) for v in xs.tolist()]
    result["y_raw"]    = [round(float(v), 4) for v in ys.tolist()]
    result["grid_n"]   = GRID

    # Save npz cache so next expand is fast
    try: _compute_and_cache_maps(meas)
    except Exception: pass
    _maps_cache[_map_key] = (result, _map_mtime)
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
        import numpy as _np

        # Use cached TDMS arrays — no re-open on every slider move
        D_arr, Z_arr, Zs_arr, n_samp = _load_tdms_arrays(meas)
        if D_arr is None:
            raise HTTPException(status_code=500, detail="Could not read TDMS channels")

        n_curves = max(1, len(D_arr) // n_samp)
        idx = max(0, min(index, n_curves - 1))
        s, e = idx * n_samp, (idx + 1) * n_samp

        def clean(arr):
            return [None if (_math.isnan(float(v)) or _math.isinf(float(v))) else round(float(v), 6)
                    for v in arr]

        d  = clean(D_arr[s:e])
        z  = clean(Z_arr[s:e])
        zs = clean(Zs_arr[s:e]) if len(Zs_arr) > e else []

        # Sensor position (cached)
        x_pos, y_pos = None, None
        try:
            xs_s, ys_s = _load_sensors(meas)
            if xs_s is not None and idx < len(xs_s):
                x_pos = round(float(xs_s[idx]), 6)
                y_pos = round(float(ys_s[idx]), 6)
        except Exception:
            pass

        return _safe_json({"index": idx, "z": z, "d": d, "zs": zs,
                           "n": len(d), "n_curves": n_curves,
                           "x_pos": x_pos, "y_pos": y_pos})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fv-maps")
def fv_maps_stream(folder: str):
    """SSE stream that processes all FV LVM files and returns topography + E maps.

    Events:
      {type:'progress', done:N, total:N, label:str}
      {type:'done', topo:[[...]], E:[[...]], grid:[rows,cols],
                    x_raw:[...], y_raw:[...],
                    x_coords:[...], y_coords:[...], grid_n:N}
      {type:'error', message:str}
    """
    meas = Path(folder)
    if not meas.exists():
        def _e():
            yield 'data: ' + _json.dumps({"type":"error","message":f"Not found: {folder}"}) + '\n\n'
        return StreamingResponse(_e(), media_type="text/event-stream")

    def ev(obj):
        return 'data: ' + _json.dumps(_sanitize(obj), ensure_ascii=False, default=str) + '\n\n'

    def generate():
        import numpy as _np
        import math as _math
        from pathlib import Path as _P

        # ── Read config ───────────────────────────────────────────────────────
        cfg_path = meas / "config.txt"
        nx, ny = 10, 10
        xlength, ylength = 0.0, 0.0
        if cfg_path.exists():
            import re as _re
            txt = cfg_path.read_text(encoding="utf-8", errors="replace")
            def _ci(key, default):
                m = _re.search(key + r',([0-9.]+)', txt)
                return int(float(m.group(1))) if m else default
            def _cf(key, default):
                m = _re.search(key + r',([0-9.]+)', txt)
                return float(m.group(1)) if m else default
            nx = _ci("Xstep", 10); ny = _ci("Ystep", 10)
            xlength = _cf("xlength", 0.0); ylength = _cf("ylength", 0.0)

        # ── Find LVM files ────────────────────────────────────────────────────
        lvm_dir = meas / "ForceCurve"
        if not lvm_dir.exists():
            lvm_dir = meas   # fallback: LVMs directly in meas folder
        import re as _re
        lvm_files = sorted(lvm_dir.glob("ForceCurve_*.lvm"),
                           key=lambda p: int(_re.search(r'(\d+)', p.stem).group(1))
                           if _re.search(r'(\d+)', p.stem) else 0)
        if not lvm_files:
            yield ev({"type":"error","message":"No ForceCurve_*.lvm files found"})
            return

        n_files = len(lvm_files)
        yield ev({"type":"progress","done":0,"total":n_files,
                  "label":f"Found {n_files} LVM files ({nx}×{ny} grid)…"})

        # ── Cantilever params from comments sidecar ───────────────────────────
        INVOLS = 0.1686; StIV = 30.0; k = 0.09; nu = 0.5; alpha = 17.5
        try:
            from afm_io import load_comments as _lc
            comments = _lc(meas)
            cant = comments.get("cantilever","AC40")
            from io_utils_PF import get_cantilever_defaults as _gcd
            cd = _gcd(cant)
            INVOLS = cd["invols"] / 1000.0   # nm→um
            k      = cd["k"]
        except Exception:
            pass

        # ── Multiprocessing ───────────────────────────────────────────────────
        n_cores = max(1, round((os.cpu_count() or 1) * 0.8))
        args_list = [(str(p), i, INVOLS, StIV, k, 8, 3, 10, nu, alpha)
                     for i, p in enumerate(lvm_files)]

        results = [None] * n_files
        done = 0
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_cores) as pool:
            for res in pool.imap_unordered(_fv_worker, args_list, chunksize=4):
                idx, zm, ds, cp, E_val = res
                results[idx] = (zm, ds, cp, E_val)
                done += 1
                yield ev({"type":"progress","done":done,"total":n_files,
                          "label":f"{lvm_files[idx].name}"})

        # ── Build maps ────────────────────────────────────────────────────────
        topo = _np.array([r[0][r[2]] if r and len(r[0]) > r[2] else _np.nan
                          for r in results], dtype=_np.float32)
        E_arr = _np.array([r[3] if r else _np.nan for r in results], dtype=_np.float32)

        # Subtract median topo baseline
        valid_t = topo[_np.isfinite(topo)]
        if len(valid_t): topo -= _np.median(valid_t)

        # Grid layout: reshape to (ny, nx)
        rows, cols = ny, nx
        topo_grid = topo[:rows*cols].reshape(rows, cols)
        E_grid    = E_arr[:rows*cols].reshape(rows, cols)

        def _mg(grid):
            flat = grid.ravel().tolist()
            vld = [v for v in flat if v is not None and _math.isfinite(v)]
            return {"data":[round(float(v),4) if _math.isfinite(float(v)) else None
                            for v in flat],
                    "rows":rows,"cols":cols,
                    "vmin":round(min(vld),4) if vld else 0,
                    "vmax":round(max(vld),4) if vld else 1}

        # X/Y positions for crosshair
        xs = _np.array([(i % cols) * (xlength / max(cols-1,1)) for i in range(rows*cols)],
                        dtype=_np.float32)
        ys = _np.array([(i // cols) * (ylength / max(rows-1,1)) for i in range(rows*cols)],
                        dtype=_np.float32)
        xi = _np.linspace(0, xlength, cols) if xlength > 0 else _np.arange(cols, dtype=_np.float32)
        yi = _np.linspace(0, ylength, rows) if ylength > 0 else _np.arange(rows, dtype=_np.float32)

        yield ev({"type":"done",
                  "topo": _mg(topo_grid),
                  "E":    _mg(E_grid),
                  "grid": [rows, cols],
                  "x_raw":    [round(float(v),3) for v in xs],
                  "y_raw":    [round(float(v),3) for v in ys],
                  "x_coords": [round(float(v),3) for v in xi],
                  "y_coords": [round(float(v),3) for v in yi],
                  "grid_n":   cols})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.get("/api/fv-fc")
def get_fv_fc(folder: str, index: int = 0):
    """Return one FV force curve by LVM file index (pre-processed)."""
    import math as _math
    meas = Path(folder)
    lvm_dir = meas / "ForceCurve"
    if not lvm_dir.exists(): lvm_dir = meas
    import re as _re
    lvm_files = sorted(lvm_dir.glob("ForceCurve_*.lvm"),
                       key=lambda p: int(_re.search(r'(\d+)', p.stem).group(1))
                       if _re.search(r'(\d+)', p.stem) else 0)
    if not lvm_files:
        raise HTTPException(status_code=404, detail="No LVM files found")

    idx = max(0, min(index, len(lvm_files)-1))
    INVOLS = 0.1686; StIV = 30.0; k = 0.09; nu = 0.5; alpha = 17.5

    res = _fv_worker((str(lvm_files[idx]), idx, INVOLS, StIV, k, 8, 3, 10, nu, alpha))
    _, zm, ds, cp, E_val = res
    nx = len(lvm_files)

    def _c(arr):
        return [None if _math.isnan(float(v)) or _math.isinf(float(v))
                else round(float(v),5) for v in arr]

    return _safe_json({
        "index": idx, "n_curves": len(lvm_files),
        "z": _c(zm), "d": _c(ds), "cp": int(cp),
        "E": None if _math.isnan(E_val) else round(E_val, 2)
    })


@app.post("/api/fv-export")
async def fv_export(request: "Request"):
    """Generate paper-ready Topo + E figure as PDF with transparent background."""
    import math as _math, io
    import numpy as _np

    body = await request.json()
    topo_data   = body.get("topo")
    e_data      = body.get("e")
    rows        = int(body.get("rows", 10))
    cols        = int(body.get("cols", 10))
    x_um        = float(body.get("x_um", 0))
    y_um        = float(body.get("y_um", 0))
    clip_pct    = float(body.get("clip_pct", 5))
    interp      = bool(body.get("interpolate", False))
    sample_name = str(body.get("sample_name", ""))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # ── helpers ──────────────────────────────────────────────────────────
        def _arr(data, r, c):
            return _np.array([v if (v is not None and _math.isfinite(v)) else _np.nan
                               for v in data], dtype=_np.float64).reshape(r, c)

        def _clip(a, pct):
            vals = a[_np.isfinite(a)]
            if not len(vals): return a, float(_np.nanmin(a)), float(_np.nanmax(a))
            return a, float(_np.percentile(vals, pct)), float(_np.percentile(vals, 100-pct))

        def _upsample_spline(a, out_rows, out_cols):
            """Sharp upsampling via 2-D cubic spline — avoids zoom artefacts."""
            from scipy.interpolate import RectBivariateSpline as _RBS
            a_f = _np.nan_to_num(a, nan=float(_np.nanmedian(a)))
            yr = _np.linspace(0, 1, a.shape[0])
            xr = _np.linspace(0, 1, a.shape[1])
            spl = _RBS(yr, xr, a_f, kx=3, ky=3)
            yo  = _np.linspace(0, 1, out_rows)
            xo  = _np.linspace(0, 1, out_cols)
            return spl(yo, xo).astype(_np.float32)

        def _nice_ticks(lo, hi, n=5):
            span = hi - lo
            if span <= 0: return [round(lo,6), round(hi,6)]
            raw  = span / (n - 1)
            mag  = 10 ** _np.floor(_np.log10(max(abs(raw), 1e-30)))
            step = round(raw / mag) * mag
            if step == 0: return [round(lo,6), round(hi,6)]
            start = _np.ceil(lo / step) * step
            t, v  = [], float(start)
            while v <= hi + 1e-9:
                t.append(round(v, 8)); v += step
            return t if len(t) >= 2 else [round(lo,6), round(hi,6)]

        T, E = _arr(topo_data, rows, cols), _arr(e_data, rows, cols)
        _, tlo, thi = _clip(T, clip_pct)
        _, elo, ehi = _clip(E, clip_pct)

        OUT = 512
        if interp:
            # Correct aspect: expand each axis proportionally
            if rows >= cols:
                out_r, out_c = OUT, max(1, round(OUT * cols / rows))
            else:
                out_r, out_c = max(1, round(OUT * rows / cols)), OUT
            T_p = _upsample_spline(T, out_r, out_c)
            E_p = _upsample_spline(E, out_r, out_c)
        else:
            T_p, E_p = T, E

        # Figure physical size: each map is 3 cm wide × correct aspect
        phys_aspect = (y_um / x_um) if x_um > 0 and y_um > 0 else (rows / max(cols, 1))
        map_w_in = 1.7   # inches per map
        map_h_in = map_w_in * phys_aspect
        map_h_in = max(0.8, min(4.0, map_h_in))   # clamp
        fig_w    = map_w_in * 2 + 1.6              # two maps + colorbars + gap
        fig_h    = map_h_in + 0.7                  # maps + title space

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        fig.patch.set_alpha(0)   # transparent background

        specs = [
            (axes[0], T_p, tlo, thi, "viridis",  "Topography",     "μm",   True),
            (axes[1], E_p, elo, ehi, "inferno",  "Elasticity (E)", "kPa",  False),
        ]
        for ax, data, vlo, vhi, cmap, title_str, unit, add_sb in specs:
            im = ax.imshow(data, cmap=cmap, vmin=vlo, vmax=vhi,
                           origin="upper",
                           interpolation="none",   # data is already upsampled
                           aspect="equal" if interp else "auto")

            # Black outline
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor("black"); sp.set_linewidth(0.7)
            ax.set_xticks([]); ax.set_yticks([])

            # Title above plot, left-aligned
            ax.set_title(title_str, fontsize=8, loc="left", pad=3, fontweight="bold")

            # Colorbar
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, aspect=18)
            cb.set_label(unit, fontsize=7, labelpad=2)
            ticks = _nice_ticks(vlo, vhi, n=5)
            cb.set_ticks([t for t in ticks if vlo <= t <= vhi] or [vlo, vhi])
            cb.ax.tick_params(labelsize=6.5)
            cb.outline.set_linewidth(0.5)

            # Scale bar (25% width, bottom-left)
            if add_sb and x_um > 0:
                h_px, w_px = data.shape
                sb_frac = 0.25
                sb_um   = x_um * sb_frac
                bar_len = w_px * sb_frac
                x0 = w_px * 0.05; y0 = h_px * 0.93
                ax.plot([x0, x0+bar_len],[y0, y0], color="white",
                        linewidth=2.0, solid_capstyle="butt", zorder=5)
                lbl = f"{sb_um:.0f} μm" if sb_um >= 1 else f"{sb_um*1000:.0f} nm"
                ax.text(x0+bar_len/2, y0-h_px*0.04, lbl,
                        color="white", fontsize=6, ha="center", va="bottom", zorder=5)

        plt.tight_layout(pad=0.5, w_pad=0.6)

        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight",
                    transparent=True, backend="pdf")
        plt.close(fig)
        buf.seek(0)
        fname = (sample_name.replace(" ","_") or "fv_maps") + ".pdf"
        return Response(content=buf.read(), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    except Exception as e:
        import traceback
        return _safe_json({"error": str(e), "traceback": traceback.format_exc()})


@app.get("/api/health")
def health():
    return _safe_json({"status": "ok", "root": _current_root})
