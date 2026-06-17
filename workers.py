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

_MAP_CACHE_VERSION = 9   # must match server.py
_E_OVERSAMPLE      = 4   # must match server.py — see note there


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

        # ── Load deflection (D) + Z from the flat LVM token stream ────────────
        # The file is a single concatenated list: all deflection samples then all
        # Z samples. Usually it's an even split (half D, half Z), but sometimes
        # the deflection channel is LONGER. The D→Z boundary is a large step in
        # value, so np.diff peaks there. We locate the split at argmax(|diff|)
        # within a window around the midpoint (the boundary is always near half),
        # but only trust it when the step clearly stands out; otherwise we fall
        # back to an even split. Two-line files are handled directly.
        _data = open(lvm_path, 'rb').read()
        _rows = [ln.split() for ln in _data.splitlines() if ln.strip()]
        if len(_rows) == 2:
            D = _np.array(_rows[0], dtype=_np.float32)
            Z = _np.array(_rows[1], dtype=_np.float32)
        else:
            raw = _np.array(_data.split(), dtype=_np.float32)
            N   = len(raw)
            half = N // 2
            if N >= 8:
                dif = _np.abs(_np.diff(raw))
                w   = max(2, N // 4)
                lo, hi = max(1, half - w), min(N - 1, half + w)
                loc = lo + int(_np.argmax(dif[lo:hi]))
                typ = float(_np.median(dif)) + 1e-12
                split = (loc + 1) if dif[loc] > 5 * typ else half
            else:
                split = half
            D = raw[:split]
            Z = raw[split:]
        if len(D) != len(Z):
            m = min(len(D), len(Z))   # trim the longer channel at the end
            D, Z = D[:m], Z[:m]

        # ── Corrected FV processing (matches PF; no phase delay — FV velocity
        #    is low so D and Z are already aligned, verified on glass data) ─────
        d = D[::sample].astype(_np.float64)
        z = (Z[::sample] * StIV).astype(_np.float64)
        n = len(d)
        if n < 8:
            return (idx, _np.array([]), _np.array([]), 0, float('nan'))

        # Z is a ramp up then down: turnaround = argmax. approach = 0..turn.
        turn = int(_np.argmax(z))
        if turn < 5 or turn > n - 5:
            turn = n // 2

        # Baseline removal: linear polyfit over the pre-contact region
        nb = max(10, turn // 3)
        try:
            base = _np.polyfit(z[:nb], d[:nb], 1)
            d_flat = (d - _np.polyval(base, z)) * INVOLS
        except Exception:
            d_flat = (d - float(_np.median(d[:nb]))) * INVOLS

        # Smooth approach deflection (stable savgol, see note above)
        appr = d_flat[:turn]
        if len(appr) >= 7:
            win = 51 if len(appr) >= 51 else (len(appr) | 1) - (0 if len(appr) % 2 else 1)
            if win > len(appr): win -= 2
            po  = min(3, max(1, win - 1))
            import warnings as _wn
            with _wn.catch_warnings():
                _wn.simplefilter("ignore")
                appr = _np.array(_sig.savgol_filter(appr, win, po), dtype=_np.float64)

        # Contact point: walk BACK from the approach deflection peak to y=0
        med = float(_np.median(appr[:nb])) if len(appr) > nb else 0.0
        sig = 1.4826 * float(_np.median(_np.abs(appr[:nb] - med))) + 1e-9 if len(appr) > nb else 1e-9
        pk  = int(_np.argmax(appr)) if len(appr) else 0
        if len(appr) and appr[pk] > med + 8*sig and pk > 3:
            cp = pk
            while cp > 0 and appr[cp] > med:
                cp -= 1
        else:
            cp = max(0, turn - 1)

        # Detachment point: adhesion minimum on the retract
        dp = turn + int(_np.argmin(d_flat[turn:])) if turn < n - 2 else n - 1

        # E via Sneddon on the contact region of the approach
        E_val = float('nan')
        if cp < turn - 3:
            Delta_m = (z[cp:turn] - z[cp]) * 1e-6
            F_N     = (d_flat[cp:turn] - d_flat[cp]) * k * 1e-6
            fmax    = int(_np.argmax(F_N)) if len(F_N) else 0
            if fmax >= 3:
                d2 = Delta_m[:fmax] ** 2; mask = d2 > 0
                if mask.sum() >= 3:
                    try:
                        alpha_r = _math.radians(alpha)
                        _x = d2[mask]; _y = F_N[:fmax][mask]
                        _dx = _x - _x.mean(); _den = float(_dx @ _dx)
                        if _den > 0:
                            sl = float(_dx @ (_y - _y.mean())) / _den
                            E_val = sl * _math.pi * (1 - nu**2) / (2 * _math.tan(alpha_r))
                            if E_val < 0: E_val = float('nan')
                    except Exception:
                        pass

        # Return the FULL curve (approach+retract) + landmarks for unfolded plot
        return (idx, z.astype(_np.float32), d_flat.astype(_np.float32),
                int(cp), E_val, int(turn), int(dp))
    except Exception:
        return (args[1], __import__('numpy').array([]), __import__('numpy').array([]),
                0, float('nan'), 0, -1)


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
            # NOTE: single backslash in these raw strings. The old code used "\\s"
            # which in a raw string is a literal backslash and matched nothing,
            # so nx/ny/sizes silently fell back to defaults (and n_samp to 500),
            # producing misaligned strided reads -> striped/garbage maps.
            m = _re.search(r"XStep:\s*([0-9]+)", txt);                 nx   = max(1,int(m.group(1))) if m else 1
            m = _re.search(r"YStep:\s*([0-9]+)", txt);                 ny   = max(1,int(m.group(1))) if m else 1
            m = _re.search(r"X.{1,10}?\(.*?m\):\s*([0-9.]+)", txt);    x_um = float(m.group(1)) if m else 0
            m = _re.search(r"Y.{1,10}?\(.*?m\):\s*([0-9.]+)", txt);    y_um = float(m.group(1)) if m else 0
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
            m = _re2.search(r"FCあたりのデータ取得点数:\s*([0-9]+)", txt)
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

    # Row-uniform curve selection — guarantees all rows are represented
    # (uniform stride leaves last N rows unsampled → edge artifacts)
    n_rows_g = max(1, n // nx if nx > 0 else 1)
    n_per_row = max(1, (GRID * GRID * _E_OVERSAMPLE) // n_rows_g)
    sel_list = []
    for _r in range(n_rows_g):
        _rs = _r * nx; _re_end = _rs + nx
        _lo = _np.searchsorted(ret_indices, _rs)
        _hi = _np.searchsorted(ret_indices, _re_end)
        _rr = ret_indices[_lo:_hi]
        if len(_rr) == 0: continue
        _st = max(1, len(_rr) // n_per_row)
        sel_list.extend(_rr[::_st][:n_per_row].tolist())
    sel = _np.array(sel_list, dtype=_np.int64)
    if len(sel) == 0:
        sel = ret_indices[::max(1,len(ret_indices)//(GRID*GRID))]
    half   = n_samp // 2
    nb     = max(5, half // 5)

    try:
        import math as _math
        from nptdms import TdmsFile as _TF
        _tdms = _TF.open(str(tdms))
        _ch_D = _tdms["Forcecurve"].channels()[0]
        _ch_Z = _tdms["Forcecurve"].channels()[1]
        # E-fit constants: INVOLS + k come from the (overridable) sidecar /
        # cantilever; nu/alpha are tip-geometry constants. Must match server.py.
        _nu = 0.5; _alpha_r = _np.radians(17.5)
        try:
            from afm_io import (resolve_invols as _ri,
                                get_cantilever_defaults as _gcd,
                                load_comments as _lc)
            _cm   = _lc(meas)
            _cant = _cm.get("cantilever")
            _INVOLS = _ri(_cm, _cant) / 1000.0   # nm/V -> um/V
            _k      = float(_gcd(_cant).get("k", 0.2))
        except Exception:
            _k = 0.2; _INVOLS = 0.1686
        cp_sel = _np.empty(len(sel), dtype=_np.float32)
        E_sel  = _np.full(len(sel), _np.nan, dtype=_np.float32)
        for out_i, fc_i in enumerate(sel):
            s = int(fc_i) * n_samp
            # Read the FULL pulse, not just the first half — in this instrument
            # the surface is reached at Z-max (~mid-pulse), so a fixed half-window
            # clips contact. turnaround = argmax(Z).
            D = _np.array(_ch_D[s:s+n_samp], dtype=_np.float32)
            Z = _np.array(_ch_Z[s:s+n_samp], dtype=_np.float32)
            turn = int(_np.argmax(Z))
            if turn < 10 or turn > n_samp - 10:
                turn = n_samp // 2
            Da, Za = D[:turn], Z[:turn]            # approach segment
            nb = max(10, turn // 3)
            # Baseline removal via linear polyfit over the pre-contact region
            try:
                base = _np.polyfit(Za[:nb], Da[:nb], 1)
                Df   = Da - _np.polyval(base, Za)
            except Exception:
                Df   = Da - float(_np.median(Da[:nb]))
            # Contact point: walk BACK from the approach deflection peak to the
            # y=0 crossing. Forward search trips on early noise spikes and
            # collapses CP to the curve start (z≈-0.5).
            med = float(_np.median(Df[:nb]))
            sig = 1.4826 * float(_np.median(_np.abs(Df[:nb] - med))) + 1e-9
            pk  = int(_np.argmax(Df))
            if Df[pk] > med + 8*sig and pk > 3:
                cp_i = pk
                while cp_i > 0 and Df[cp_i] > med:
                    cp_i -= 1
            else:
                cp_i = turn - 1
            cp_sel[out_i] = Z[cp_i]
            # E via Sneddon fit on the contact region (baseline-removed deflection)
            if cp_i < turn - 3:
                Dc = (Df[cp_i:] - Df[cp_i]) * _INVOLS  # um deflection
                Zc = (Za[cp_i:] - Za[cp_i])            # um z-travel
                delta = (Zc - Dc) * 1e-6               # indentation [m]
                F     = Dc * _k * 1e-6                 # force [N]
                fm    = int(_np.argmax(F))
                if fm >= 3:
                    x = delta[:fm] ** 2; y = F[:fm]
                    mk = x > 0
                    if mk.sum() >= 3:
                        x = x[mk]; y = y[mk]
                        dx = x - x.mean(); den = float(dx @ dx)
                        if den > 0:
                            sl = float(dx @ (y - y.mean())) / den
                            Ev = sl * _math.pi * (1 - _nu**2) / (2 * _math.tan(_alpha_r))
                            if Ev > 0: E_sel[out_i] = Ev
    except Exception:
        return False

    xs_sel = xs_r[_np.searchsorted(ret_indices, sel)]
    ys_sel = ys_r[_np.searchsorted(ret_indices, sel)]
    qpts   = _np.column_stack([Xi.ravel(), Yi.ravel()])
    tree   = _KDT(_np.column_stack([xs_sel, ys_sel]))
    _, idx = tree.query(qpts, workers=-1)
    cp_grid = cp_sel[idx].reshape(GRID, GRID)
    E_grid  = E_sel[idx].reshape(GRID, GRID)
    g2f     = sel[idx].astype(_np.int32)

    arrays = dict(e=E_grid.astype(_np.float32),
                  xi=xi.astype(_np.float32), yi=yi.astype(_np.float32),
                  x_raw=xs, y_raw=ys, grid_to_fc=g2f,
                  grid_n=_np.array([GRID], dtype=_np.int32),
                  e_invols=_np.array([_INVOLS], dtype=_np.float32),
                  _version=_np.array([_MAP_CACHE_VERSION], dtype=_np.int32))

    # Load ZTipoffsets (for topo) and ZSamplePID (displayed separately as requested).
    ztip_grid = None
    for fname, key in [("ZTipoffsets.txt","ztip"), ("ZSamplePID.txt","zpid")]:
        p = meas / fname
        if p.exists():
            try:
                v = _np.array(open(str(p),"rb").read().split(), dtype=_np.float32)
                if len(v) == n:
                    v_sel = v[rmask][_np.searchsorted(ret_indices, sel)]
                    grid  = v_sel[idx].reshape(GRID, GRID)
                    if key == "ztip":
                        ztip_grid = grid
                    else:
                        arrays[key] = grid
            except Exception: pass

    # Topography = ZTipoffsets (slow PID Z) + CP (contact-point fine Z),
    # matching afm_drag.py fig_topomap: topo = offset + cp_h.
    # Median-subtract so the map is flat-referenced.
    topo_grid = (ztip_grid + cp_grid) if ztip_grid is not None else cp_grid.copy()
    _med = float(_np.nanmedian(topo_grid))
    if _np.isfinite(_med):
        topo_grid -= _med
    arrays["topo"] = topo_grid.astype(_np.float32)
    try:
        _np.savez_compressed(str(npz), **arrays)
        return True
    except Exception:
        return False
