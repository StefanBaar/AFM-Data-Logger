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
import time
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




# ── Local import ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
import workers as _workers   # minimal-import worker module (Windows-safe spawn)
from afm_io import (discover_all, update_dataset_meta,
                     _find_measurement_configs, _find_fv_configs,
                     parse_pf_path, parse_fv_path,
                     parse_pf_config, parse_fv_config,
                     load_comments, scan_tdms_stage,
                     _dataset_is_filled, _create_nan_config)



# Bump this whenever map computation logic changes — forces all npz to regenerate
_MAP_CACHE_VERSION = 9

# Curve oversampling for nearest-neighbour map gridding. Each grid cell takes
# exactly one nearest sample, so this mainly trades read+fit time for spatial
# fidelity. 4 = original (smoothest). Lower = faster (2 roughly halves curves
# read/fit for a barely-visible fidelity change at GRID>=50). Min sensible: 1.
_E_OVERSAMPLE = 4

# Bump when FV map computation changes — forces afm_fv_maps.npz to regenerate.
_FV_CACHE_VERSION = 1

_sensor_cache: dict = {}   # folder -> (xs, ys)
_maps_cache:   dict = {}   # folder -> (result_dict, mtime)

# ── Background pre-processing state ────────────────────────────────────────────
# A single background job pre-computes afm_maps.npz for every PF folder so that
# expanding a row is instant. Status is pollable from the UI.
_prewarm_lock   = threading.Lock()
_prewarm_state  = {"running": False, "done": 0, "total": 0,
                   "current": "", "started": 0.0, "finished": 0.0,
                   "errors": 0}

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


def _get_n_samp(meas: "Path") -> int:
    """Read samples-per-FC from config.txt."""
    import re as _re
    cfg = Path(meas) / "config.txt"
    if cfg.exists():
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
            m = _re.search(u"FC\u3042\u305f\u308a\u306e\u30c7\u30fc\u30bf\u53d6\u5f97\u70b9\u6570:\\s*([0-9]+)", txt)
            if m: return int(m.group(1))
        except Exception: pass
    return 500


def _get_y_size_um(meas: "Path") -> float:
    """Read Y measurement size (um) from config.txt. Returns 0.0 if absent/NaN —
    callers use 0.0 as the signal to render a square-aspect map."""
    import re as _re
    cfg = Path(meas) / "config.txt"
    if cfg.exists():
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
            m = _re.search(r"Y.{1,10}?\(.*?m\):\s*([0-9.]+)", txt)
            if m: return float(m.group(1))
        except Exception: pass
    return 0.0


def _process_fc(d_raw: "np.ndarray", z_raw: "np.ndarray", meas: "Path"):
    """Full single-curve processing, validated against glass reference data.

    Returns dict with: z (phase-fit clean Z), d (delay-aligned, baseline-removed
    deflection), turn, cp, dp (detachment), zunf (unfolded Z).

    Steps (matching afm_drag.py + the glass-slope-match calibration):
      1. Fit cosine drive model to raw Z  →  clean z_theo (removes sensor noise).
         turn = argmax(z_theo): Z-max is closest approach (the surface is
         reached as Z increases in this instrument).
      2. Find the deflection/Z time lag by matching the post-contact slope of
         approach and retract (on glass they must be equal). Shift D to align.
      3. Remove a tilted baseline with a linear polyfit over the pre-contact
         (off-surface) region → flat zero baseline before contact.
      4. Contact point = where the rising approach deflection crosses y=0.
      5. Detachment point = adhesion minimum on the retract.
    """
    import numpy as _np
    n = len(z_raw)
    try:
        import re as _re
        cfg = (meas / "config.txt").read_text(encoding="utf-8", errors="replace")
        def _f(pat, dflt):
            m = _re.search(pat, cfg); return float(m.group(1)) if m else dflt
        ph0 = _f(r"データ取得開始位相:\s*([0-9.]+)", 0.0)
        ph1 = _f(r"データ取得終了位相:\s*([0-9.]+)", 0.99)
    except Exception:
        ph0, ph1 = 0.0, 0.99

    ph = _np.linspace(ph0, ph1, n)
    G  = _np.column_stack([_np.cos(2*_np.pi*ph), _np.sin(2*_np.pi*ph), _np.ones(n)])
    try:
        a, b, c = _np.linalg.lstsq(G, z_raw.astype(_np.float64), rcond=None)[0]
        A   = float(_np.hypot(a, b)); phi = float(_np.arctan2(-b, a))
        z_theo = (A*_np.cos(2*_np.pi*ph + phi) + c).astype(_np.float32)
    except Exception:
        z_theo = z_raw.astype(_np.float32)

    turn = int(_np.argmax(z_theo))
    if turn < 5 or turn > n-5:
        turn = n // 2

    d = d_raw.astype(_np.float64)

    # ── Deflection delay: match approach/retract contact slope (glass) ────────
    def _slope_gap(delay):
        if delay <= 0: return 1e9
        Dd = d[delay:]; Zd = z_theo[:n-delay]
        tn = int(_np.argmax(Zd))
        aD, aZ = Dd[:tn], Zd[:tn]; rD, rZ = Dd[tn:], Zd[tn:]
        thr = 0.3 * float(_np.nanmax(d))
        ca = aD > thr; cr = rD > thr
        if ca.sum() < 5 or cr.sum() < 5: return 1e9
        try:
            sa = _np.polyfit(aZ[ca], aD[ca], 1)[0]
            sr = _np.polyfit(rZ[cr], rD[cr], 1)[0]
            return abs(sa - sr)
        except Exception:
            return 1e9
    best_delay, best_gap = 0, 1e9
    for dl in range(0, 14):
        g = _slope_gap(dl)
        if g < best_gap: best_gap, best_delay = g, dl
    # Pad the tail so length is preserved
    if best_delay > 0:
        d = _np.concatenate([d[best_delay:], _np.full(best_delay, d[-1])])

    # ── Baseline removal: linear polyfit over pre-contact region ──────────────
    nb = max(10, turn // 3)
    try:
        base = _np.polyfit(z_theo[:nb], d[:nb], 1)
        d_flat = d - _np.polyval(base, z_theo)
    except Exception:
        d_flat = d - float(_np.median(d[:nb]))

    # ── Contact point: walk BACK from the deflection peak to the y=0 crossing ─
    # Searching forward from the start is fragile — a single noise spike near
    # sample 0 trips the threshold and the CP collapses to the bottom of the
    # curve (z≈-0.5). Contact is always just before the turnaround, so we start
    # at the approach deflection maximum and descend until D returns to baseline
    # (crosses zero). That crossing is the contact point.
    cp = -1
    fa = d_flat[:turn]
    if len(fa) > nb + 3:
        med = float(_np.median(fa[:nb]))
        sig = 1.4826 * float(_np.median(_np.abs(fa[:nb] - med))) + 1e-9
        pk  = int(_np.argmax(fa))
        # Require a real contact (peak well above baseline noise)
        if fa[pk] > med + 8.0 * sig and pk > 3:
            j = pk
            while j > 0 and fa[j] > med:
                j -= 1
            cp = j

    # ── Detachment point: adhesion minimum on retract ─────────────────────────
    dp = -1
    if turn < n - 2:
        dp = turn + int(_np.argmin(d_flat[turn:]))

    # ── Unfolded Z (cumulative |dz|, turnaround = 1.0) ────────────────────────
    zunf = _np.concatenate([[0.0], _np.cumsum(_np.abs(_np.diff(z_theo)))])
    norm = zunf[turn] if zunf[turn] > 0 else 1.0
    zunf = (zunf / norm).astype(_np.float32)

    return {"z": z_theo, "d": d_flat.astype(_np.float32), "zunf": zunf,
            "turn": turn, "cp": cp, "dp": dp, "delay": best_delay}


def _phase_correct_z(z_raw: "np.ndarray", meas: "Path"):
    """Legacy wrapper — returns (z_theo, turn=argmax). Kept for compatibility."""
    import numpy as _np
    n = len(z_raw)
    try:
        import re as _re
        cfg = (meas / "config.txt").read_text(encoding="utf-8", errors="replace")
        def _f(pat, dflt):
            m = _re.search(pat, cfg); return float(m.group(1)) if m else dflt
        ph0 = _f(r"データ取得開始位相:\s*([0-9.]+)", 0.0)
        ph1 = _f(r"データ取得終了位相:\s*([0-9.]+)", 0.99)
        ph = _np.linspace(ph0, ph1, n)
        G  = _np.column_stack([_np.cos(2*_np.pi*ph), _np.sin(2*_np.pi*ph), _np.ones(n)])
        a, b, c = _np.linalg.lstsq(G, z_raw.astype(_np.float64), rcond=None)[0]
        A = float(_np.hypot(a, b)); phi = float(_np.arctan2(-b, a))
        z_theo = (A*_np.cos(2*_np.pi*ph + phi) + c).astype(_np.float32)
        return z_theo, int(_np.argmax(z_theo))
    except Exception:
        return z_raw, len(z_raw) // 2


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
    """Compute maps for preview using strided TDMS reads.

    Only reads GRID×GRID evenly-spaced curves from the TDMS file — never
    the full file.  For a 1000×1000 scan we read 100 curves (~400KB) not
    1M curves (3.7GB).  Returns True if a fresh valid cache exists or was
    just written.
    """
    import numpy as _np
    from scipy.spatial import cKDTree as _KDT
    from pathlib import Path as _P
    import re as _re
    meas = _P(meas)
    npz  = meas / "afm_maps.npz"
    tdms = meas / "ForceCurve.tdms"
    if not tdms.exists(): return False

    # Skip if cache already fresh + correct version
    if npz.exists() and npz.stat().st_mtime >= tdms.stat().st_mtime:
        try:
            v = _np.load(str(npz))
            if int(v.get("_version", _np.array([0]))[0]) == _MAP_CACHE_VERSION:
                return True   # already good — don't touch TDMS
        except Exception:
            pass

    xs, ys = _load_sensors(meas)
    if xs is None: return False
    n_pts  = len(xs)
    n_samp = _get_n_samp(meas)
    if n_samp == 0: return False

    GRID   = min(100, max(10, int(_np.sqrt(n_pts))))
    xi     = _np.linspace(xs.min(), xs.max(), GRID)
    yi     = _np.linspace(ys.min(), ys.max(), GRID)
    Xi, Yi = _np.meshgrid(xi, yi)

    # Return-direction mask
    dx    = _np.diff(xs, prepend=xs[0])
    rmask = dx < 0
    if rmask.sum() < 4: rmask = _np.ones(n_pts, dtype=bool)
    xs_r, ys_r = xs[rmask], ys[rmask]

    ret_indices = _np.where(rmask)[0]  # FC indices that are return-direction

    # Get nx (columns per row) from config for row-uniform sampling
    nx = round(_np.sqrt(n_pts))  # fallback: square grid
    cfg = meas / "config.txt"
    if cfg.exists():
        try:
            _m = _re.search(r"XStep:\s*([0-9]+)",
                            cfg.read_text(encoding="utf-8", errors="replace"))
            if _m: nx = max(1, int(_m.group(1)))
        except Exception: pass
    if nx * 2 > n_pts: nx = max(1, round(_np.sqrt(n_pts)))

    # Row-uniform curve selection — guarantees all rows are represented
    # (uniform stride leaves last N rows unsampled → edge artifacts)
    n_rows_g = max(1, n_pts // nx)
    n_per_row = max(1, (GRID * GRID * _E_OVERSAMPLE) // n_rows_g)
    sel_list = []
    for _r in range(n_rows_g):
        _rs = _r * nx; _re = _rs + nx
        _lo = _np.searchsorted(ret_indices, _rs)
        _hi = _np.searchsorted(ret_indices, _re)
        _rr = ret_indices[_lo:_hi]
        if len(_rr) == 0: continue
        _st = max(1, len(_rr) // n_per_row)
        sel_list.extend(_rr[::_st][:n_per_row].tolist())
    sel = _np.array(sel_list, dtype=_np.int64)
    if len(sel) == 0:
        sel = ret_indices[::max(1,len(ret_indices)//(GRID*GRID))]


    # Read only the selected curves from TDMS
    half = n_samp // 2
    nb   = max(5, half // 5)
    try:
        from nptdms import TdmsFile as _TF
        _tdms = _TF.open(str(tdms))  # lazy open — no full read
        _ch_D = _tdms["Forcecurve"].channels()[0]
        _ch_Z = _tdms["Forcecurve"].channels()[1]
        cp_sel = _np.empty(len(sel), dtype=_np.float32)
        for out_i, fc_i in enumerate(sel):
            s = int(fc_i) * n_samp
            D = _np.array(_ch_D[s:s+half], dtype=_np.float32)
            Z = _np.array(_ch_Z[s:s+half], dtype=_np.float32)
            bm = D[:nb].mean(); bs = max(float(D[:nb].std()), 1e-6)
            above = _np.where(D > bm + 5*bs)[0]
            cp_i  = int(above[0]) if len(above) else half - 1
            if cp_i == 0 and len(above) == 0: cp_i = half - 1  # no contact
            cp_sel[out_i] = Z[cp_i]
    except Exception:
        return False

    # ── Grid using only selected points ──────────────────────────────────────
    xs_sel = xs_r[_np.searchsorted(ret_indices, sel)]  # positions of sel curves
    ys_sel = ys_r[_np.searchsorted(ret_indices, sel)]
    qpts   = _np.column_stack([Xi.ravel(), Yi.ravel()])
    tree   = _KDT(_np.column_stack([xs_sel, ys_sel]))
    _, idx = tree.query(qpts, workers=-1)
    cp_grid = cp_sel[idx].reshape(GRID, GRID)

    # grid_to_fc: each grid cell → the actual FC index in the full scan
    g2f = sel[idx].astype(_np.int32)

    # ── E map via Sneddon fit on contact region (same strided curves) ─────────
    _k = 0.2; _nu = 0.5; _alpha_r = _np.radians(17.5); _INVOLS = 0.1686
    E_sel = _np.full(len(sel), _np.nan, dtype=_np.float32)
    try:
        from nptdms import TdmsFile as _TF2
        _ch_D2 = _TF2.open(str(tdms))["Forcecurve"].channels()[0]
        _ch_Z2 = _TF2.open(str(tdms))["Forcecurve"].channels()[1]
        for _oi, _fci in enumerate(sel):
            _s = int(_fci) * n_samp
            _D = _np.array(_ch_D2[_s:_s+half], dtype=_np.float32)
            _Z = _np.array(_ch_Z2[_s:_s+half], dtype=_np.float32)
            _bm = _D[:nb].mean(); _bs = max(float(_D[:nb].std()), 1e-6)
            _ab = _np.where(_D > _bm + 5*_bs)[0]
            _cp = int(_ab[0]) if len(_ab) else half-1
            if _cp >= half - 3: continue
            _Dc = (_D[_cp:] - _D[_cp]) * _INVOLS   # um deflection
            _Zc = _Z[_cp:] - _Z[_cp]               # um z-travel
            _delta = (_Zc - _Dc) * 1e-6             # indentation [m]
            _F     = _Dc * _k * 1e-6                # force [N]
            _fm    = int(_np.argmax(_F))
            if _fm < 3: continue
            _d2 = _delta[:_fm]**2; _mk = _d2 > 0
            if _mk.sum() < 3: continue
            try:
                _sl, _ = _np.polyfit(_d2[_mk], _F[:_fm][_mk], 1)
                _E = float(_sl * _np.pi * (1-_nu**2) / (2*_np.tan(_alpha_r)))
                if _E > 0: E_sel[_oi] = _E
            except Exception: pass
    except Exception: pass
    E_grid = E_sel[idx].reshape(GRID, GRID)

    arrays = dict(cp=cp_grid.astype(_np.float32),
                  e=E_grid.astype(_np.float32),
                  xi=xi.astype(_np.float32), yi=yi.astype(_np.float32),
                  x_raw=xs, y_raw=ys,
                  grid_to_fc=g2f,
                  grid_n=_np.array([GRID], dtype=_np.int32),
                  _version=_np.array([_MAP_CACHE_VERSION], dtype=_np.int32))

    # Sensor text maps (ZSamplePID, ZTipoffsets) — these are cheap txt files
    for fname, key in [("ZSamplePID.txt","zpid"), ("ZTipoffsets.txt","ztip")]:
        p = meas / fname
        if p.exists():
            try:
                v = _fast_load_txt(p).astype(_np.float32)
                if len(v) == n_pts:
                    v_sel = v[rmask][_np.searchsorted(ret_indices, sel)]
                    arrays[key] = v_sel[idx].reshape(GRID, GRID)
            except Exception: pass

    try:
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
                        _workers.stage_worker, pf_folders_to_scan, chunksize=1):
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


@app.post("/api/prewarm")
def prewarm_start(force: bool = False):
    """Kick off background pre-computation of all PF map caches (afm_maps.npz).

    Non-blocking: returns immediately. Poll /api/prewarm-status for progress.
    Folders whose cache is already fresh are skipped (instant), unless force=1.
    Uses a process pool so each TDMS read/E-fit runs off the main event loop.
    """
    with _prewarm_lock:
        if _prewarm_state["running"]:
            return _safe_json({"ok": True, "already_running": True,
                               **_prewarm_state})
        if not _current_root:
            return _safe_json({"ok": False, "error": "no_root"})

        try:
            datasets = discover_all(Path(_current_root))
        except Exception as e:
            return _safe_json({"ok": False, "error": str(e)})

        folders = []
        for d in datasets:
            if d.get("mode") != "PF":
                continue
            f = Path(d["folder"])
            tdms = f / "ForceCurve.tdms"
            if not tdms.exists():
                continue
            if not force:
                npz = f / "afm_maps.npz"
                # Skip if cache is newer than the TDMS (fresh)
                if npz.exists() and npz.stat().st_mtime >= tdms.stat().st_mtime:
                    continue
            folders.append(str(f))

        _prewarm_state.update(running=True, done=0, total=len(folders),
                              current="", started=time.time(), finished=0.0,
                              errors=0)

    if not folders:
        with _prewarm_lock:
            _prewarm_state.update(running=False, finished=time.time())
        return _safe_json({"ok": True, "total": 0, "note": "all caches fresh"})

    def _runner(folder_list):
        ctx = multiprocessing.get_context("spawn")
        n_cores = max(1, round((os.cpu_count() or 1) * 0.8))
        try:
            with ctx.Pool(processes=n_cores) as pool:
                for folder_str, result in pool.imap_unordered(
                        _workers.stage_worker, folder_list, chunksize=1):
                    with _prewarm_lock:
                        _prewarm_state["done"] += 1
                        _prewarm_state["current"] = Path(folder_str).name
                        if not result:
                            _prewarm_state["errors"] += 1
        except Exception:
            with _prewarm_lock:
                _prewarm_state["errors"] += 1
        finally:
            with _prewarm_lock:
                _prewarm_state["running"] = False
                _prewarm_state["finished"] = time.time()
                _prewarm_state["current"] = ""

    threading.Thread(target=_runner, args=(folders,), daemon=True).start()
    return _safe_json({"ok": True, "total": len(folders)})


@app.get("/api/prewarm-status")
def prewarm_status():
    with _prewarm_lock:
        s = dict(_prewarm_state)
    if s["total"]:
        s["pct"] = round(s["done"] / s["total"] * 100)
    else:
        s["pct"] = 100
    if s["started"] and s["finished"]:
        s["elapsed"] = round(s["finished"] - s["started"], 1)
    elif s["started"]:
        s["elapsed"] = round(time.time() - s["started"], 1)
    return _safe_json(s)


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
    """SSE stream: sends progress during TDMS processing, then the full map result."""
    import json as _json

    meas = Path(folder)
    if not meas.exists():
        def _err():
            yield 'data: ' + _json.dumps({"type":"error","message":f"Folder not found: {folder}"}) + '\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    def ev(obj):
        return 'data: ' + _json.dumps(_sanitize(obj), ensure_ascii=False, default=str) + '\n\n'

    def generate():
        import math as _math, numpy as _np, numpy as _np2, time as _time

        # ── RAM cache hit (instant) ───────────────────────────────────────────
        _map_mtime = 0
        for _fn in ["ForceCurve.tdms","Xsensors.txt","ZSamplePID.txt","ZTipoffsets.txt"]:
            _fp = meas / _fn
            if _fp.exists(): _map_mtime = max(_map_mtime, _fp.stat().st_mtime)
        _map_key = str(meas)
        if _map_key in _maps_cache and _maps_cache[_map_key][1] == _map_mtime:
            yield ev({"type":"done", **_maps_cache[_map_key][0]})
            return

        result = {"folder": folder, "maps": {}, "n_curves": None,
                  "x_coords": None, "y_coords": None}

        # ── Fast path: load pre-computed npz ─────────────────────────────────
        _npz = meas / "afm_maps.npz"
        _tdms2 = meas / "ForceCurve.tdms"
        if _npz.exists() and (not _tdms2.exists() or _npz.stat().st_mtime >= _tdms2.stat().st_mtime):
            try:
                npz = _np2.load(str(_npz))
                if int(npz.get("_version", _np2.array([0]))[0]) != _MAP_CACHE_VERSION:
                    raise ValueError("stale")
                # Defensive: a PF folder (has TDMS) must have an E map. If a cache
                # was written without it (e.g. by an older/partial writer), treat
                # it as stale and fall through to recompute rather than silently
                # serving a result with no E panel.
                if _tdms2.exists() and "e" not in npz.files:
                    raise ValueError("missing E map")
                # Invalidate if INVOLS changed since this E map was computed —
                # the user can edit INVOLS per-measurement and expects E to follow.
                if _tdms2.exists():
                    try:
                        from afm_io import resolve_invols as _ri, load_comments as _lc
                        _cm = _lc(meas)
                        _want = round(_ri(_cm, _cm.get("cantilever")) / 1000.0, 6)
                        _have = round(float(npz["e_invols"][0]), 6) if "e_invols" in npz.files else None
                        if _have is None or abs(_have - _want) > 1e-9:
                            raise ValueError("INVOLS changed")
                    except ValueError:
                        raise
                    except Exception:
                        pass
                GRID2 = int(npz["grid_n"][0])
                def _nm(arr):
                    flat=[round(float(v),4) for v in arr.ravel()]
                    return {"data":flat,"rows":GRID2,"cols":GRID2,"n":len(flat),
                            "vmin":round(float(_np2.nanmin(arr)),4),
                            "vmax":round(float(_np2.nanmax(arr)),4)}
                if "topo" in npz.files:
                    result["maps"]["Topo"] = _nm(npz["topo"])
                elif "cp" in npz.files:
                    # Legacy cache (pre-v7): compute topo on the fly
                    _cp_g = npz["cp"]
                    _topo = (_cp_g + npz["ztip"]) if "ztip" in npz.files else _cp_g.copy()
                    _med  = float(_np2.nanmedian(_topo))
                    if _np2.isfinite(_med): _topo -= _med
                    result["maps"]["Topo"] = _nm(_topo.astype(_np2.float32))
                if "zpid" in npz.files:
                    result["maps"]["ZSamplePID"] = _nm(npz["zpid"])
                if "e" in npz.files:
                    result["maps"]["E"]    = _nm((npz["e"]/1000.0).astype(_np2.float32))
                result["n_curves"] = len(npz["x_raw"])
                result["x_coords"] = [round(float(v),4) for v in npz["xi"].tolist()]
                result["y_coords"] = [round(float(v),4) for v in npz["yi"].tolist()]
                result["grid_n"]   = GRID2
                result["y_size_um"] = _get_y_size_um(meas)
                if "grid_to_fc" in npz.files:
                    result["grid_to_fc"] = npz["grid_to_fc"].tolist()
                _maps_cache[_map_key] = (result, _map_mtime)
                yield ev({"type":"done", **result})
                return
            except Exception:
                pass

        # ── Slow path: read TDMS and compute ─────────────────────────────────
        # Stream progress during the TDMS curve reads
        yield ev({"type":"progress", "pct":5, "label":"Loading sensor files…"})

        xs, ys = _load_sensors(meas)
        if xs is None:
            yield ev({"type":"done", **result}); return

        n_pts  = len(xs)
        n_samp = _get_n_samp(meas)
        GRID   = min(100, max(10, int(_np.sqrt(n_pts))))
        xi     = _np.linspace(xs.min(), xs.max(), GRID)
        yi     = _np.linspace(ys.min(), ys.max(), GRID)
        Xi, Yi = _np.meshgrid(xi, yi)
        dx     = _np.diff(xs, prepend=xs[0])
        rmask  = dx < 0
        if rmask.sum() < 4: rmask = _np.ones(n_pts, dtype=bool)
        xs_r, ys_r = xs[rmask], ys[rmask]
        ret_indices = _np.where(rmask)[0]

        # Read nx from config for row-uniform sampling
        import re as _re2
        _nx = 1
        _cfg = meas / "config.txt"
        if _cfg.exists():
            _m = _re2.search(r"XStep:\s*([0-9]+)",
                             _cfg.read_text(encoding="utf-8", errors="replace"))
            if _m: _nx = max(1, int(_m.group(1)))
        if _nx * 2 > n_pts: _nx = max(1, round(_np.sqrt(n_pts)))

        # Row-uniform curve selection — guarantees all rows are represented
        # (uniform stride leaves last N rows unsampled → edge artifacts)
        _n_rows_g = max(1, n_pts // _nx)
        _n_per_row = max(1, (GRID * GRID * _E_OVERSAMPLE) // _n_rows_g)
        _sel_list = []
        for _r in range(_n_rows_g):
            _rs = _r * _nx; _re = _rs + _nx
            _lo = _np.searchsorted(ret_indices, _rs)
            _hi = _np.searchsorted(ret_indices, _re)
            _rr = ret_indices[_lo:_hi]
            if len(_rr) == 0: continue
            _st = max(1, len(_rr) // _n_per_row)
            _sel_list.extend(_rr[::_st][:_n_per_row].tolist())
        sel = _np.array(_sel_list, dtype=_np.int64)
        if len(sel) == 0:
            sel = ret_indices[::max(1, len(ret_indices)//(GRID*GRID))]
        half   = n_samp // 2
        nb     = max(5, half // 5)
        n_sel  = len(sel)

        yield ev({"type":"progress", "pct":10,
                  "label":f"Reading {n_sel} curves from TDMS…",
                  "done":0, "total":n_sel})

        cp_sel = _np.empty(n_sel, dtype=_np.float32)
        E_sel  = _np.full(n_sel, _np.nan, dtype=_np.float32)
        # E-fit constants: INVOLS + k come from the (overridable) sidecar /
        # cantilever; nu/alpha are tip-geometry constants.
        _nu=0.5; _alpha_r=_np.radians(17.5)
        try:
            from afm_io import resolve_invols as _ri, get_cantilever_defaults as _gcd, load_comments as _lc
            _cm   = _lc(meas)
            _cant = _cm.get("cantilever")
            _INVOLS = _ri(_cm, _cant) / 1000.0   # nm/V -> um/V
            _k      = float(_gcd(_cant).get("k", 0.2))
        except Exception:
            _k = 0.2; _INVOLS = 0.1686

        try:
            from nptdms import TdmsFile as _TF
            _tdms = _TF.open(str(_tdms2))
            _chD  = _tdms["Forcecurve"].channels()[0]
            _chZ  = _tdms["Forcecurve"].channels()[1]
            import math as _math
            REPORT_EVERY = max(1, n_sel // 20)  # ~20 progress updates

            for _oi, _fci in enumerate(sel):
                _s = int(_fci) * n_samp
                _D = _np.array(_chD[_s:_s+half], dtype=_np.float32)
                _Z = _np.array(_chZ[_s:_s+half], dtype=_np.float32)
                _bm = _D[:nb].mean(); _bs = max(float(_D[:nb].std()), 1e-6)
                _ab = _np.where(_D > _bm + 5*_bs)[0]
                _cp = int(_ab[0]) if len(_ab) else half-1
                cp_sel[_oi] = _Z[_cp]
                # E via Sneddon
                if _cp < half - 3:
                    _Dc=(_D[_cp:]-_D[_cp])*_INVOLS; _Zc=_Z[_cp:]-_Z[_cp]
                    _delta=(_Zc-_Dc)*1e-6; _F=_Dc*_k*1e-6
                    _fm=int(_np.argmax(_F))
                    if _fm>=3:
                        _x=_delta[:_fm]**2; _y=_F[:_fm]; _mk=_x>0
                        if _mk.sum()>=3:
                            _x=_x[_mk]; _y=_y[_mk]
                            _dx=_x-_x.mean(); _den=float(_dx@_dx)
                            if _den>0:
                                _sl=float(_dx@(_y-_y.mean()))/_den
                                _Ev=_sl*_math.pi*(1-_nu**2)/(2*_math.tan(_alpha_r))
                                if _Ev>0: E_sel[_oi]=_Ev

                if (_oi+1) % REPORT_EVERY == 0 or _oi == n_sel-1:
                    pct = 10 + round((_oi+1)/n_sel * 70)
                    yield ev({"type":"progress","pct":pct,
                              "label":f"Processing curves… {_oi+1}/{n_sel}",
                              "done":_oi+1,"total":n_sel})
        except Exception as _ex:
            yield ev({"type":"progress","pct":80,"label":f"TDMS error: {_ex}"})

        yield ev({"type":"progress","pct":82,"label":"Building grid…"})

        # Grid
        from scipy.spatial import cKDTree as _KDT
        xs_sel = xs_r[_np.searchsorted(ret_indices, sel)]
        ys_sel = ys_r[_np.searchsorted(ret_indices, sel)]
        qpts   = _np.column_stack([Xi.ravel(), Yi.ravel()])
        tree   = _KDT(_np.column_stack([xs_sel, ys_sel]))
        _, idx = tree.query(qpts, workers=-1)
        cp_grid = cp_sel[idx].reshape(GRID, GRID)
        E_grid  = E_sel[idx].reshape(GRID, GRID)
        g2f     = sel[idx].astype(_np.int32)

        arrays = dict(cp=cp_grid.astype(_np.float32),
                      e=E_grid.astype(_np.float32),
                      xi=xi.astype(_np.float32), yi=yi.astype(_np.float32),
                      x_raw=xs, y_raw=ys, grid_to_fc=g2f,
                      grid_n=_np.array([GRID], dtype=_np.int32),
                      e_invols=_np.array([_INVOLS], dtype=_np.float32),
                      _version=_np.array([_MAP_CACHE_VERSION], dtype=_np.int32))

        for fname, key in [("ZSamplePID.txt","zpid"), ("ZTipoffsets.txt","ztip")]:
            p = meas / fname
            if p.exists():
                try:
                    v = _fast_load_txt(p).astype(_np.float32)
                    if len(v) == n_pts:
                        v_sel = v[rmask][_np.searchsorted(ret_indices, sel)]
                        arrays[key] = v_sel[idx].reshape(GRID, GRID)
                except Exception: pass

        yield ev({"type":"progress","pct":92,"label":"Saving cache…"})
        try:
            _np.savez_compressed(str(_npz), **arrays)
        except Exception: pass

        yield ev({"type":"progress","pct":96,"label":"Building response…"})

        def _nm3(arr):
            flat=[round(float(v),4) for v in arr.ravel()]
            return {"data":flat,"rows":GRID,"cols":GRID,"n":len(flat),
                    "vmin":round(float(_np.nanmin(arr)),4),
                    "vmax":round(float(_np.nanmax(arr)),4)}
        # Topo = ZTipoffsets (slow PID Z) + CP (fine contact height), median-subtracted.
        # Matches afm_drag.py fig_topomap: topo = offset + cp_h.
        _ztip = arrays.get("ztip")
        _topo = (_ztip + cp_grid) if _ztip is not None else cp_grid.copy()
        _med  = float(_np.nanmedian(_topo[_np.isfinite(_topo)]))
        if _np.isfinite(_med): _topo -= _med
        arrays["topo"] = _topo.astype(_np.float32)
        # Drop the individual components — they're absorbed into topo
        arrays.pop("cp",   None)
        arrays.pop("ztip", None)
        # zpid (ZSamplePID) stays in arrays — saved to cache and shown as its own map

        result["maps"]["Topo"] = _nm3(_topo)
        if "zpid" in arrays: result["maps"]["ZSamplePID"] = _nm3(arrays["zpid"])
        result["maps"]["E"]    = _nm3(E_grid/1000.0)
        result["n_curves"]  = n_pts
        result["x_coords"]  = [round(float(v),4) for v in xi.tolist()]
        result["y_coords"]  = [round(float(v),4) for v in yi.tolist()]
        result["grid_n"]    = GRID
        result["y_size_um"] = _get_y_size_um(meas)
        result["grid_to_fc"]= g2f.tolist()

        if result.get("maps"):
            _maps_cache[_map_key] = (result, _map_mtime)
        yield ev({"type":"done", **result})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

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

        n_samp = _get_n_samp(meas)

        def clean(arr):
            return [None if (_math.isnan(float(v)) or _math.isinf(float(v))) else round(float(v), 6)
                    for v in arr]

        # Try cache first (fast for small/medium scans already loaded)
        D_arr, Z_arr, Zs_arr, n_samp_c = _load_tdms_arrays(meas)
        if D_arr is not None:
            n_samp   = n_samp_c
            n_curves = max(1, len(D_arr) // n_samp)
            idx      = max(0, min(index, n_curves - 1))
            s, e     = idx * n_samp, (idx + 1) * n_samp
            d  = clean(D_arr[s:e])
            z  = clean(Z_arr[s:e])
            zs = clean(Zs_arr[s:e]) if len(Zs_arr) > e else []
        else:
            # Large file: lazy open, slice just one curve — no full array load
            from nptdms import TdmsFile as _TF
            tdms     = _TF.open(str(tdms_path))
            chs      = tdms["Forcecurve"].channels()
            n_total  = len(chs[0])
            n_curves = max(1, n_total // n_samp)
            idx      = max(0, min(index, n_curves - 1))
            s, e     = idx * n_samp, (idx + 1) * n_samp
            d  = clean(_np.array(chs[0][s:e], dtype=_np.float32))
            z  = clean(_np.array(chs[1][s:e], dtype=_np.float32))
            zs = clean(_np.array(chs[2][s:e], dtype=_np.float32)) if len(chs) > 2 else []

        # Sensor position (cached)
        x_pos, y_pos = None, None
        try:
            xs_s, ys_s = _load_sensors(meas)
            if xs_s is not None and idx < len(xs_s):
                x_pos = round(float(xs_s[idx]), 6)
                y_pos = round(float(ys_s[idx]), 6)
        except Exception:
            pass

        # Full processing: phase-fit Z, delay-align D, baseline removal,
        # CP at y=0 crossing, detachment point, unfolded Z. Validated on glass.
        _z_raw = _np.array([v if v is not None else 0.0 for v in z], dtype=_np.float64)
        _d_raw = _np.array([v if v is not None else 0.0 for v in d], dtype=_np.float64)
        _r = _process_fc(_d_raw, _z_raw, meas)

        return _safe_json({"index": idx, "n": _r["turn"], "n_curves": n_curves,
                           "z":    clean(_r["z"]),
                           "d":    clean(_r["d"]),
                           "zunf": clean(_r["zunf"]),
                           "zs":   zs,
                           "x_pos": x_pos, "y_pos": y_pos,
                           "turn": _r["turn"], "cp": _r["cp"], "dp": _r["dp"]})
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
            from afm_io import (load_comments as _lc,
                                resolve_invols as _ri,
                                get_cantilever_defaults as _gcd)
            comments = _lc(meas)
            cant = comments.get("cantilever") or "AC40"
            INVOLS = _ri(comments, cant) / 1000.0   # nm/V -> um/V (honours override)
            cd     = _gcd(cant)
            k      = cd.get("k", k)
            nu     = cd.get("nu", nu)
            alpha  = cd.get("alpha", alpha)
        except Exception:
            pass

        # ── Fast path: serve a fresh cache instead of reprocessing ────────────
        # The old code reprocessed every LVM on every expand. Now, if a cached
        # afm_fv_maps.npz is newer than all source files, matches the cache
        # version, and was computed with the current INVOLS, emit it instantly.
        _npz_fv = meas / "afm_fv_maps.npz"
        try:
            _src_mtime = max([p.stat().st_mtime for p in lvm_files] +
                             [cfg_path.stat().st_mtime if cfg_path.exists() else 0])
            if _npz_fv.exists() and _npz_fv.stat().st_mtime >= _src_mtime:
                _v = _np.load(str(_npz_fv))
                _ver_ok = int(_v.get("_version", _np.array([0]))[0]) == _FV_CACHE_VERSION
                _inv_c  = float(_v["e_invols"][0]) if "e_invols" in _v.files else None
                if _ver_ok and _inv_c is not None and abs(_inv_c - INVOLS) < 1e-9:
                    _tg = _v["topo"]; _eg = _v["E"]
                    _gr = _v["grid"]; _rc, _cc = int(_gr[0]), int(_gr[1])
                    _xi = _v["xi"]; _yi = _v["yi"]; _xr = _v["x_raw"]; _yr = _v["y_raw"]
                    _part = bool(_v["partial"][0]) if "partial" in _v.files else False
                    def _mgc(grid):
                        flat = grid.ravel().tolist()
                        vld = [x for x in flat if x is not None and _math.isfinite(float(x))]
                        return {"data":[round(float(x),4) if _math.isfinite(float(x)) else None
                                        for x in flat],
                                "rows":_rc,"cols":_cc,
                                "vmin":round(min(vld),4) if vld else 0,
                                "vmax":round(max(vld),4) if vld else 1}
                    yield ev({"type":"progress","done":n_files,"total":n_files,
                              "label":"Loaded from cache \u2713"})
                    yield ev({"type":"done",
                              "topo": _mgc(_tg), "E": _mgc(_eg), "grid":[_rc,_cc],
                              "x_raw":[round(float(x),3) for x in _xr],
                              "y_raw":[round(float(x),3) for x in _yr],
                              "x_coords":[round(float(x),3) for x in _xi],
                              "y_coords":[round(float(x),3) for x in _yi],
                              "grid_n":_cc, "partial":_part,
                              "x_um":xlength, "y_um":ylength})
                    return
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
            for res in pool.imap_unordered(_workers.fv_worker, args_list, chunksize=4):
                idx, zm, ds, cp, E_val = res[0], res[1], res[2], res[3], res[4]
                results[idx] = (zm, ds, cp, E_val)
                done += 1
                yield ev({"type":"progress","done":done,"total":n_files,
                          "label":f"{lvm_files[idx].name}"})

        # ── Build maps ────────────────────────────────────────────────────────
        # topo from contact-point Z height; E from Sneddon fit.
        topo = _np.array([r[0][r[2]] if r and len(r[0]) > r[2] else _np.nan
                          for r in results], dtype=_np.float32)
        E_arr = _np.array([r[3] if r else _np.nan for r in results], dtype=_np.float32)

        # Subtract median topo baseline
        valid_t = topo[_np.isfinite(topo)]
        if len(valid_t): topo -= _np.median(valid_t)

        # ── Handle complete vs incomplete scans ──────────────────────────────
        cols         = nx
        is_complete  = (n_files >= nx * ny)

        if is_complete:
            rows        = ny
            n_data_rows = ny   # all rows are data rows
            topo_grid   = topo[:rows*cols].reshape(rows, cols)
            E_grid      = E_arr[:rows*cols].reshape(rows, cols)
        else:
            # Interrupted scan: keep data rows + 3 empty padding rows
            n_complete_rows = n_files // cols
            n_partial       = n_files % cols
            n_data_rows     = n_complete_rows + (1 if n_partial > 0 else 0)
            EXTRA_ROWS      = 3
            rows            = n_data_rows + EXTRA_ROWS
            total_cells     = rows * cols
            topo_pad        = _np.full(total_cells, _np.nan, dtype=_np.float32)
            E_pad           = _np.full(total_cells, _np.nan, dtype=_np.float32)
            topo_pad[:n_files] = topo[:n_files]
            E_pad[:n_files]    = E_arr[:n_files]
            topo_grid = topo_pad.reshape(rows, cols)
            E_grid    = E_pad.reshape(rows, cols)

        def _mg(grid):
            flat = grid.ravel().tolist()
            vld = [v for v in flat if v is not None and _math.isfinite(v)]
            return {"data":[round(float(v),4) if _math.isfinite(float(v)) else None
                            for v in flat],
                    "rows":rows,"cols":cols,
                    "vmin":round(min(vld),4) if vld else 0,
                    "vmax":round(max(vld),4) if vld else 1}

        # X/Y positions for crosshair — only for actual data points
        xs = _np.array([(i % cols) * (xlength / max(cols-1,1)) for i in range(rows*cols)],
                        dtype=_np.float32)
        ys = _np.array([(i // cols) * (ylength / max(rows-1,1)) for i in range(rows*cols)],
                        dtype=_np.float32)
        # y coords span only the data rows (not the padding)
        y_data_end = ylength * n_data_rows / max(ny, 1) if ylength > 0 else n_data_rows
        y_pad_end  = y_data_end * rows / max(n_data_rows, 1)
        xi = _np.linspace(0, xlength,   cols) if xlength > 0 else _np.arange(cols, dtype=_np.float32)
        yi = _np.linspace(0, y_pad_end, rows) if ylength > 0 else _np.arange(rows, dtype=_np.float32)

        # ── Save full-resolution cache next to config ─────────────────────
        _npz_fv = meas / "afm_fv_maps.npz"
        try:
            _np.savez_compressed(str(_npz_fv),
                topo=topo_grid, E=E_grid,
                xi=xi.astype(_np.float32), yi=yi.astype(_np.float32),
                x_raw=xs.astype(_np.float32), y_raw=ys.astype(_np.float32),
                xlength=_np.array([xlength], dtype=_np.float32),
                ylength=_np.array([ylength], dtype=_np.float32),
                grid=_np.array([rows, cols], dtype=_np.int32),
                partial=_np.array([0 if is_complete else 1], dtype=_np.int32),
                e_invols=_np.array([INVOLS], dtype=_np.float32),
                _version=_np.array([_FV_CACHE_VERSION], dtype=_np.int32))
        except Exception: pass

        yield ev({"type":"done",
                  "topo": _mg(topo_grid),
                  "E":    _mg(E_grid),
                  "grid": [rows, cols],
                  "x_raw":    [round(float(v),3) for v in xs],
                  "y_raw":    [round(float(v),3) for v in ys],
                  "x_coords": [round(float(v),3) for v in xi],
                  "y_coords": [round(float(v),3) for v in yi],
                  "grid_n":   cols,
                  "partial":  not is_complete,
                  "x_um": xlength, "y_um": ylength})

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
    try:
        from afm_io import (load_comments as _lc,
                            resolve_invols as _ri,
                            get_cantilever_defaults as _gcd)
        _cm   = _lc(meas)
        _cant = _cm.get("cantilever") or "AC40"
        INVOLS = _ri(_cm, _cant) / 1000.0
        _cd   = _gcd(_cant)
        k = _cd.get("k", k); nu = _cd.get("nu", nu); alpha = _cd.get("alpha", alpha)
    except Exception:
        pass

    res = _workers.fv_worker((str(lvm_files[idx]), idx, INVOLS, StIV, k, 8, 3, 10, nu, alpha))
    idx_r, zm, ds, cp, E_val = res[0], res[1], res[2], res[3], res[4]
    turn = int(res[5]) if len(res) > 5 else len(zm) - 1
    dp   = int(res[6]) if len(res) > 6 else -1

    def _c(arr):
        return [None if _math.isnan(float(v)) or _math.isinf(float(v))
                else round(float(v),5) for v in arr]

    # Full curve (approach+retract). Unfolded Z = cumulative |dz| normalised so
    # the turnaround = 1.0 (approach 0..1, retract 1..2), like the PF view.
    import numpy as _np
    _zm = _np.asarray(zm, dtype=_np.float64)
    if len(_zm) > 1 and 0 < turn < len(_zm):
        _zunf = _np.concatenate([[0.0], _np.cumsum(_np.abs(_np.diff(_zm)))])
        _norm = _zunf[turn] if _zunf[turn] > 0 else 1.0
        _zunf = _zunf / _norm
    else:
        _zunf = _np.zeros(len(_zm))

    return _safe_json({
        "index": idx, "n_curves": len(lvm_files),
        "z": _c(zm), "d": _c(ds), "cp": int(cp),
        "zunf": _c(_zunf), "turn": turn, "dp": dp,
        "E": None if _math.isnan(E_val) else round(E_val, 2)
    })


def _make_pdf(specs, x_um, y_um, clip_pct, interp, title=None):
    """Shared PDF renderer for PF and FV maps.
    specs = list of (name, data_2d, unit, cmap)
    Returns bytes of a transparent-bg PDF.
    """
    import io as _io, math as _math
    import numpy as _np
    import matplotlib as _mpl; _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    n = len(specs)
    phys_aspect = (y_um / x_um) if x_um > 0 and y_um > 0 else 1.0

    def _clip(a, pct):
        v = a[_np.isfinite(a)]
        if not len(v): return 0.0, 1.0
        return float(_np.percentile(v, pct)), float(_np.percentile(v, 100-pct))

    def _upsample(a, out_r, out_c):
        from scipy.interpolate import RectBivariateSpline as _S
        af = _np.nan_to_num(a, nan=float(_np.nanmedian(a)))
        s  = _S(_np.linspace(0,1,a.shape[0]), _np.linspace(0,1,a.shape[1]), af, kx=3, ky=3)
        return s(_np.linspace(0,1,out_r), _np.linspace(0,1,out_c)).astype(_np.float32)

    def _nice_ticks(lo, hi, n=5):
        span = hi - lo
        if span <= 0: return [round(lo,6), round(hi,6)]
        raw = span / (n-1)
        mag = 10 ** _np.floor(_np.log10(max(abs(raw), 1e-30)))
        step = round(raw / mag) * mag or mag
        start = _np.ceil(lo / step) * step
        t = []; v = float(start)
        while v <= hi + 1e-9: t.append(round(v,8)); v += step
        return t if len(t) >= 2 else [round(lo,6), round(hi,6)]

    # Prepare images
    imgs = []
    for name, arr, unit, cmap in specs:
        lo, hi = _clip(arr, clip_pct)
        if interp:
            OUT = 1024
            or_ = OUT if phys_aspect >= 1 else max(1, round(OUT * phys_aspect))
            oc_ = OUT if phys_aspect <  1 else max(1, round(OUT / phys_aspect))
            arr = _upsample(arr, or_, oc_)
            # Clip E (kPa) to non-negative — cubic spline can overshoot
            if unit in ("kPa", "Pa"): arr = _np.clip(arr, 0.0, None)
            imethod = "bilinear"
        else:
            imethod = "nearest"
        imgs.append((name, arr, unit, cmap, lo, hi, imethod))

    map_w = 1.6
    map_h = max(0.8, min(5.0, map_w * phys_aspect))
    fig_w = n * (map_w + 0.5) + 0.3
    fig_h = map_h + 0.5

    fig, axes = _plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1: axes = [axes]
    fig.patch.set_alpha(0)

    ext = [0, x_um if x_um > 0 else 1, y_um if y_um > 0 else 1, 0]

    for i, (ax, (name, arr, unit, cmap, lo, hi, imethod)) in enumerate(zip(axes, imgs)):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi,
                       origin="upper", extent=ext,
                       interpolation=imethod, aspect="equal")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor("black"); sp.set_linewidth(0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=8, loc="left", pad=3, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, aspect=18)
        cb.set_label(unit, fontsize=7, labelpad=2)
        tks = [t for t in _nice_ticks(lo, hi) if lo <= t <= hi] or [lo, hi]
        cb.set_ticks(tks); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
        # Scale bar on first map only
        if i == 0 and x_um > 0:
            sb_um = x_um * 0.25
            x0 = x_um * 0.04
            y0 = (y_um if y_um > 0 else 1) * 0.93
            ax.plot([x0, x0+sb_um], [y0, y0], color="white",
                    linewidth=2.0, solid_capstyle="butt", zorder=5)
            lbl = f"{sb_um:.0f} μm" if sb_um >= 1 else f"{sb_um*1000:.0f} nm"
            ax.text(x0+sb_um/2, y0-(y_um if y_um>0 else 1)*0.05, lbl,
                    color="white", fontsize=6, ha="center", va="bottom", zorder=5)

    _plt.tight_layout(pad=0.4, w_pad=0.3)
    buf = _io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight",
                transparent=True, dpi=150)
    _plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.post("/api/fv-export")
async def fv_export(request: "Request"):
    """Paper-ready PDF of FV Topography + E maps."""
    import math as _m, numpy as _np
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
        def _a(d): return _np.array(
            [v if (v is not None and _m.isfinite(v)) else _np.nan for v in d],
            dtype=_np.float64).reshape(rows, cols)
        T = _a(topo_data); E = _a(e_data)
        specs = [("Topography", T, "μm", "viridis"),
                 ("Elasticity (E)", E/1000, "kPa", "inferno")]
        pdf = _make_pdf(specs, x_um, y_um, clip_pct, interp)
        fname = (sample_name.replace(" ","_") or "fv_maps") + ".pdf"
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})
    except Exception as e:
        import traceback
        return _safe_json({"error": str(e), "traceback": traceback.format_exc()})


@app.post("/api/pf-export")
async def pf_export(request: "Request"):
    """Paper-ready PDF of PF maps (Topo, E)."""
    import math as _m, numpy as _np
    body = await request.json()
    maps_data   = body.get("maps", {})
    x_um        = float(body.get("x_um", 0))
    y_um        = float(body.get("y_um", 0))
    clip_pct    = float(body.get("clip_pct", 5))
    interp      = bool(body.get("interpolate", False))
    try:
        cmaps = ["cividis", "inferno", "viridis", "plasma", "magma"]
        units    = {"Topo":"V", "ZSamplePID":"V", "E":"kPa"}
        cmap_for = {"Topo":"cividis", "ZSamplePID":"viridis", "E":"inferno"}
        specs = []
        for i, (nm, md) in enumerate(maps_data.items()):
            a = _np.array(
                [v if (v is not None and _m.isfinite(v)) else _np.nan for v in md["data"]],
                dtype=_np.float64).reshape(md["rows"], md["cols"])
            label = nm.replace(".txt","")
            specs.append((label, a, units.get(nm, "V"), cmap_for.get(nm, cmaps[i % len(cmaps)])))
        if not specs:
            return _safe_json({"error": "no maps"})
        pdf = _make_pdf(specs, x_um, y_um, clip_pct, interp)
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="pf_maps.pdf"'})
    except Exception as e:
        import traceback
        return _safe_json({"error": str(e), "traceback": traceback.format_exc()})


@app.get("/api/fv-recompute")
def fv_recompute(folder: str):
    """Delete afm_fv_maps.npz so the next expand reprocesses from scratch."""
    meas = Path(folder)
    npz  = meas / "afm_fv_maps.npz"
    deleted = False
    if npz.exists():
        try: npz.unlink(); deleted = True
        except Exception as e: return _safe_json({"ok": False, "error": str(e)})
    return _safe_json({"ok": True, "deleted": deleted})


@app.get("/api/health")
def health():
    return _safe_json({"status": "ok", "root": _current_root})


# Windows: prevent recursive spawning when running as frozen exe
if __name__ == '__main__':
    multiprocessing.freeze_support()
