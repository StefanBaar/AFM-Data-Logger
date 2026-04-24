"""
workers.py — Multiprocessing worker functions.

Kept separate from server.py so that on Windows (spawn start method)
each worker process only imports numpy/scipy/nptdms — NOT FastAPI or the
full server stack. This cuts per-worker RAM from ~200MB down to ~40MB.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure afm_io is importable from worker processes
sys.path.insert(0, str(Path(__file__).parent))

_MAP_CACHE_VERSION = 3   # must match server.py


# ── FV LVM worker ─────────────────────────────────────────────────────────────

def fv_worker(args) -> tuple:
    """Process one FV LVM file: load → pre_process → Sneddon fit.
    args = (lvm_path, idx, INVOLS, StIV, k, ran, th, sample, nu, alpha)
    Returns (idx, zm_f32, ds_f32, cp_int, E_float)
    """
    try:
        import numpy as _np
        from scipy import signal as _sig
        import math as _math

        lvm_path, idx, INVOLS, StIV, k, ran, th, sample, nu, alpha = args

        raw = open(lvm_path, 'rb').read().split()
        n = len(raw) // 2
        D = _np.array(raw[:n],    dtype=_np.float32)
        Z = _np.array(raw[n:2*n], dtype=_np.float32)

        d, z = D[::sample], Z[::sample]
        z = z * StIV
        base_n = max(1, len(d) // ran)
        try:    slope = float(_np.polyfit(z[:base_n], d[:base_n], 1)[0])
        except: slope = 0.0
        d = d - slope * z
        d = (d - float(_np.median(d[:base_n]))) * INVOLS
        sig = float(d[:base_n].std()) * th
        dmax = int(_np.argmax(d))
        dm, zm = d[:dmax], z[:dmax]

        if len(dm) >= 53:
            ds = _np.array(_sig.savgol_filter(dm, 51, 11), dtype=_np.float32)
        else:
            ds = dm.copy()

        below = _np.argwhere(ds < sig)
        cp = int(below[-1][0]) if len(below) else max(0, len(zm) - 1)

        E_val = float('nan')
        if cp < len(zm) - 3:
            Delta_m = (zm[cp:] - zm[cp]) * 1e-6
            F_N     = (ds[cp:] - ds[cp]) * k * 1e-6
            fmax    = int(_np.argmax(F_N))
            if fmax >= 3:
                d2   = Delta_m[:fmax] ** 2
                mask = d2 > 0
                if mask.sum() >= 3:
                    try:
                        alpha_r = _math.radians(alpha)
                        sl, _   = _np.polyfit(d2[mask], F_N[:fmax][mask], 1)
                        E_val   = float(sl * _math.pi * (1 - nu**2) /
                                        (2 * _math.tan(alpha_r)))
                        if E_val < 0: E_val = float('nan')
                    except Exception:
                        pass

        return (idx, zm.astype(_np.float32), ds.astype(_np.float32), cp, E_val)
    except Exception:
        return (args[1], __import__('numpy').array([]), __import__('numpy').array([]), 0, float('nan'))


# ── PF stage scan + map pre-computation worker ────────────────────────────────

def stage_worker(folder_str: str) -> tuple:
    """Run TDMS stage scan + pre-compute afm_maps.npz for one PF folder."""
    try:
        from afm_io import scan_tdms_stage as _scan
        result = _scan(Path(folder_str))
        try:    _compute_maps(folder_str)
        except Exception: pass
        return (folder_str, result)
    except Exception:
        return (folder_str, {})


def _kdt_grid(pts, vals, Xi, Yi, GRID):
    """Nearest-neighbour grid using cKDTree with all CPU cores."""
    import numpy as _np
    from scipy.spatial import cKDTree as _KDT
    src = _np.column_stack([pts[0], pts[1]])
    qry = _np.column_stack([Xi.ravel(), Yi.ravel()])
    _, idx = _KDT(src).query(qry, workers=-1)
    return vals[idx].reshape(GRID, GRID).astype(_np.float32)


def _compute_maps(folder_str: str) -> bool:
    """Strided TDMS map pre-computation for worker processes.
    Reads only GRID×GRID curves — never the full file.
    Skips immediately if a fresh valid npz already exists.
    """
    import numpy as _np
    from scipy.spatial import cKDTree as _KDT
    import re as _re

    meas = Path(folder_str)
    npz  = meas / "afm_maps.npz"
    tdms = meas / "ForceCurve.tdms"
    if not tdms.exists(): return False

    # Skip if already fresh
    if npz.exists() and npz.stat().st_mtime >= tdms.stat().st_mtime:
        try:
            v = _np.load(str(npz))
            if int(v.get("_version", _np.array([0]))[0]) == _MAP_CACHE_VERSION:
                return True
        except Exception:
            pass

    # Load X sensor + synthesise Y
    xf = meas / "Xsensors.txt"
    if not xf.exists(): return False
    xs = _np.array(open(str(xf), "rb").read().split(), dtype=_np.float32)
    n  = len(xs)
    if n == 0: return False

    nx, ny, x_um, y_um = 1, 1, 0.0, 0.0
    cfg = meas / "config.txt"
    if cfg.exists():
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
            for pat, var in [(r"XStep:\\s*([0-9]+)", "nx"), (r"YStep:\\s*([0-9]+)", "ny"),
                             (r"X.{1,10}?\(.*?m\):\\s*([0-9.]+)", "x_um"),
                             (r"Y.{1,10}?\(.*?m\):\\s*([0-9.]+)", "y_um")]:
                m = _re.search(pat, txt)
                if m: locals()[var]  # just check it exists
            m = _re.search(r"XStep:\\s*([0-9]+)", txt);  nx = max(1,int(m.group(1))) if m else 1
            m = _re.search(r"YStep:\\s*([0-9]+)", txt);  ny = max(1,int(m.group(1))) if m else 1
            m = _re.search(r"X.{1,10}?\(.*?m\):\\s*([0-9.]+)", txt); x_um=float(m.group(1)) if m else 0
            m = _re.search(r"Y.{1,10}?\(.*?m\):\\s*([0-9.]+)", txt); y_um=float(m.group(1)) if m else 0
        except Exception: pass
    if nx * ny != n:
        nx = max(1, round(float(_np.sqrt(n)))); ny = max(1, n // nx)

    x_range = float(xs.max() - xs.min()) or 1.0
    y_range  = x_range * (y_um / x_um) if x_um > 0 and y_um > 0 else x_range
    row_idx  = _np.arange(n) // nx
    ys = (xs.min() + row_idx * y_range / max(ny-1, 1)).astype(_np.float32)

    n_samp = 500
    if cfg.exists():
        try:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
            import re as _re2
            m = _re2.search("FCあたりのデータ取得点数:\\s*([0-9]+)", txt)
            if m: n_samp = int(m.group(1))
        except: pass

    # Return mask
    dx    = _np.diff(xs, prepend=xs[0])
    rmask = dx < 0
    if rmask.sum() < 4: rmask = _np.ones(n, dtype=bool)
    xs_r, ys_r = xs[rmask], ys[rmask]
    ret_indices = _np.where(rmask)[0]

    GRID   = min(100, max(10, int(_np.sqrt(n))))
    xi     = _np.linspace(xs.min(), xs.max(), GRID)
    yi     = _np.linspace(ys.min(), ys.max(), GRID)
    Xi, Yi = _np.meshgrid(xi, yi)

    # Strided curve selection
    n_ret  = len(ret_indices)
    n_want = min(n_ret, GRID * GRID * 4)
    stride = max(1, n_ret // n_want)
    sel    = ret_indices[::stride][:n_want]
    half   = n_samp // 2
    nb     = max(5, half // 5)

    try:
        from nptdms import TdmsFile as _TF
        _tdms = _TF.open(str(tdms))
        _ch_D = _tdms["Forcecurve"].channels()[0]
        _ch_Z = _tdms["Forcecurve"].channels()[1]
        cp_sel = _np.empty(len(sel), dtype=_np.float32)
        for out_i, fc_i in enumerate(sel):
            s = int(fc_i) * n_samp
            D = _np.array(_ch_D[s:s+half], dtype=_np.float32)
            Z = _np.array(_ch_Z[s:s+half], dtype=_np.float32)
            bm = D[:nb].mean(); bs = max(float(D[:nb].std()), 1e-6)
            above = _np.where(D > bm + 5*bs)[0]
            cp_sel[out_i] = Z[int(above[0])] if len(above) else Z[half-1]
    except Exception:
        return False

    xs_sel = xs_r[_np.searchsorted(ret_indices, sel)]
    ys_sel = ys_r[_np.searchsorted(ret_indices, sel)]
    qpts   = _np.column_stack([Xi.ravel(), Yi.ravel()])
    tree   = _KDT(_np.column_stack([xs_sel, ys_sel]))
    _, idx = tree.query(qpts, workers=-1)
    cp_grid = cp_sel[idx].reshape(GRID, GRID)
    g2f     = sel[idx].astype(_np.int32)

    arrays = dict(cp=cp_grid.astype(_np.float32),
                  xi=xi.astype(_np.float32), yi=yi.astype(_np.float32),
                  x_raw=xs, y_raw=ys, grid_to_fc=g2f,
                  grid_n=_np.array([GRID], dtype=_np.int32),
                  _version=_np.array([_MAP_CACHE_VERSION], dtype=_np.int32))

    for fname, key in [("ZSamplePID.txt","zpid"), ("ZTipoffsets.txt","ztip")]:
        p = meas / fname
        if p.exists():
            try:
                v = _np.array(open(str(p),"rb").read().split(), dtype=_np.float32)
                if len(v) == n:
                    v_sel = v[rmask][_np.searchsorted(ret_indices, sel)]
                    arrays[key] = v_sel[idx].reshape(GRID, GRID)
            except Exception: pass
    try:
        _np.savez_compressed(str(npz), **arrays)
        return True
    except Exception:
        return False
