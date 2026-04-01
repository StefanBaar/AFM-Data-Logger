"""
afm_io.py — Unified AFM data I/O for Force Volume (FV) and Pulse Force (PF) modes.
Handles config parsing, path discovery, and metadata extraction for both formats.
"""

from __future__ import annotations

import re
import json
import shutil
from pathlib import Path
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

CANTILEVER_DEFAULTS = {
    "AC40":  dict(invols=140.0, k=0.09,  nu=0.5, alpha=17.5),
    "AC160": dict(invols=140.0, k=26.0,  nu=0.5, alpha=17.5),
    "AC240": dict(invols=140.0, k=2.0,   nu=0.5, alpha=17.5),
}

ENCODINGS = ["shift_jis", "cp932", "utf-8", "utf-8-sig", "latin-1"]


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> Optional[str]:
    """Try multiple encodings; return text or None."""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return None


def _extract(pattern: str, text: str, cast=float) -> Optional[object]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, TypeError):
        return None


def get_cantilever_defaults(name: str) -> dict:
    upper = (name or "").upper().strip()
    for prefix, defaults in CANTILEVER_DEFAULTS.items():
        if upper.startswith(prefix):
            return defaults.copy()
    return CANTILEVER_DEFAULTS["AC40"].copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Comments sidecar  (.afm_comments.json next to config.txt)
# ─────────────────────────────────────────────────────────────────────────────

def _comments_path(config_dir: Path) -> Path:
    return config_dir / ".afm_comments.json"


def load_comments(config_dir: Path) -> dict:
    p = _comments_path(config_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_comments(config_dir: Path, data: dict) -> None:
    _comments_path(config_dir).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

PF_NAN_CONFIG = """FCあたりのデータ取得点数: NaN
XStep:  NaN
YStep:  NaN
X計測範囲(μm):  NaN
Y計測範囲(μm):  NaN
周波数(Hz):  NaN
データ取得開始位相:  NaN
データ取得終了位相:  NaN
ZSample gain(P, I, D):  NaN, NaN, NaN
振幅(V):  NaN
トリガ電圧(V):  NaN
空振り時PID係数:  NaN
CP閾値Deflection(V):  NaN
ZSample gain(P, I, D):  NaN, NaN, NaN
カンチレバー種類:  NaN
"""

FV_NAN_CONFIG = """start_time,NaN
end_time,NaN
Vtrig,NaN
Zig,NaN
num_app,NaN
num_ret,NaN
xlength,NaN
ylength,NaN
filter,NaN
app_speed_ratio,NaN
ret_speed_ratio,NaN
Xstep,NaN
Ystep,NaN
loop_time,NaN
FIFO_loop,NaN
ret_length,NaN
"""


def _create_nan_config(config_path: Path, mode: str) -> None:
    """Write a stub config.txt with NaN placeholders if none exists."""
    if config_path.exists():
        return
    template = PF_NAN_CONFIG if mode == "PF" else FV_NAN_CONFIG
    try:
        config_path.write_text(template, encoding="utf-8")
    except Exception:
        pass  # non-fatal — sidecar still saves


def _dataset_is_filled(cfg_data: dict, comments: dict) -> bool:
    """Return True if the dataset has at least one real user-supplied value."""
    # Check comments sidecar for any filled field
    if comments.get("cantilever") or comments.get("comments") or comments.get("sample_name"):
        return True
    # Check if config has any non-None, non-NaN numeric value
    for v in cfg_data.values():
        if v is not None:
            return True
    return False




# ─────────────────────────────────────────────────────────────────────────────
#  Pulse Force config parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_pf_config(config_path: Path) -> dict:
    """
    Parse a PF config.txt.
    Returns dict with keys matching the display table columns.
    All values are None if the file is missing or unreadable.
    """
    text = _read_text(config_path) if config_path.exists() else None
    if text is None:
        return _empty_pf()

    r = {}
    r["n_samples"]    = _extract(r"FCあたりのデータ取得点数:\s*([0-9]+)", text, int)
    r["x_step"]       = _extract(r"XStep:\s*([0-9.]+)", text, float)
    r["y_step"]       = _extract(r"YStep:\s*([0-9.]+)", text, float)
    r["x_length"]     = _extract(r"X計測範囲\(μm\):\s*([0-9.]+)", text, float)
    r["y_length"]     = _extract(r"Y計測範囲\(μm\):\s*([0-9.]+)", text, float)
    r["frequency_hz"] = _extract(r"周波数\(Hz\):\s*([0-9.]+)", text, float)
    r["u_amplitude"]  = _extract(r"振幅\(V\):\s*([0-9.]+)", text, float)
    r["u_trigger"]    = _extract(r"トリガ電圧\(V\):\s*([0-9.]+)", text, float)
    r["phase_start"]  = _extract(r"データ取得開始位相:\s*([0-9.]+)", text, float)
    r["phase_end"]    = _extract(r"データ取得終了位相:\s*([0-9.]+)", text, float)
    r["cantilever"]   = _extract(r"カンチレバー種類:\s*(\S+)", text, str)
    return r


def _empty_pf() -> dict:
    return {k: None for k in [
        "n_samples", "x_step", "y_step", "x_length", "y_length",
        "frequency_hz", "u_amplitude", "u_trigger",
        "phase_start", "phase_end", "cantilever",
    ]}


# ─────────────────────────────────────────────────────────────────────────────
#  Force Volume config parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fv_csv(text: str) -> dict:
    """Parse comma-separated key,value config format."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            k, _, v = line.partition(",")
            result[k.strip()] = v.strip()
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def parse_fv_config(config_path: Path) -> dict:
    """
    Parse a FV config.txt.
    Returns dict with keys matching the display table columns.
    """
    text = _read_text(config_path) if config_path.exists() else None
    if text is None:
        return _empty_fv()

    raw = _parse_fv_csv(text)

    def _f(key): return _safe_float(raw.get(key))
    def _i(key): return _safe_int(raw.get(key))
    def _s(key): return raw.get(key)

    r = {}
    r["start_time"]    = _s("start_time")
    r["end_time"]      = _s("end_time")
    r["u_trigger"]     = _f("Vtrig")
    r["num_approach"]  = _i("num_app")
    r["num_retract"]   = _i("num_ret")
    r["x_length"]      = _f("xlength")
    r["y_length"]      = _f("ylength")
    r["x_step"]        = _i("Xstep")
    r["y_step"]        = _i("Ystep")
    r["loop_time"]     = _f("loop_time")
    r["ret_length"]    = _f("ret_length")
    r["app_speed"]     = _f("app_speed_ratio")
    r["ret_speed"]     = _f("ret_speed_ratio")
    r["filter"]        = _s("filter")
    r["cantilever"]    = None   # not in FV config; comes from sidecar
    # Derived velocity: speed_ratio * ret_length(um)*1000 / loop_time(ms) -> nm/s
    try:
        r["velocity_app"] = round(float(raw["app_speed_ratio"]) * float(raw["ret_length"]) * 1000 / float(raw["loop_time"]), 1)
        r["velocity_ret"] = round(float(raw["ret_speed_ratio"]) * float(raw["ret_length"]) * 1000 / float(raw["loop_time"]), 1)
    except Exception:
        r["velocity_app"] = None
        r["velocity_ret"] = None
    return r


def _empty_fv() -> dict:
    return {k: None for k in [
        "start_time", "end_time", "u_trigger",
        "num_approach", "num_retract",
        "x_length", "y_length", "x_step", "y_step",
        "loop_time", "ret_length", "app_speed", "ret_speed",
        "filter", "cantilever", "velocity_app", "velocity_ret",
    ]}


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Path parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_pf_path(config_path: Path) -> dict:
    """
    PF layout: ROOT/PF/YYMMDD/HHMM_n/config.txt
    Time folder is HHMM (4 digits) + optional _n suffix.
    Date folder is YYMMDD (6 digits).
    """
    parts = config_path.parts
    info = {"date": None, "time": None, "meas_id": None, "folder": config_path.parent}

    if len(parts) >= 3:
        date_folder = parts[-3]
        meas_folder = parts[-2]

        # date: YYMMDD → 20YY-MM-DD
        m = re.match(r"^(\d{2})(\d{2})(\d{2})$", date_folder)
        if m:
            info["date"] = f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"

        # time+id: HHMM_n  or  HHMMSS_n  (handle both 4 and 6 digit times)
        m = re.match(r"^(\d{2})(\d{2})(\d{2})?(?:_(.+))?$", meas_folder)
        if m:
            hh, mm = m.group(1), m.group(2)
            ss = m.group(3) or "00"
            info["time"] = f"{hh}:{mm}:{ss}"
            info["meas_id"] = m.group(4) or ""

    return info


def parse_fv_path(config_path: Path) -> dict:
    """
    FV layout: ROOT/FV/YYYYMMDD/SAMPLENAME/HHMMSS/config.txt
    Date folder is YYYYMMDD (8 digits) or YYMMDD (6 digits) — handles both.
    Time folder is HHMMSS (6 digits).
    """
    parts = config_path.parts
    info = {
        "date": None, "time": None, "sample_name": None,
        "folder": config_path.parent
    }

    if len(parts) >= 4:
        date_folder   = parts[-4]
        sample_folder = parts[-3]
        time_folder   = parts[-2]

        # YYYYMMDD (8 digits)
        m = re.match(r"^(\d{4})(\d{2})(\d{2})$", date_folder)
        if m:
            info["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        else:
            # YYMMDD (6 digits) fallback
            m = re.match(r"^(\d{2})(\d{2})(\d{2})$", date_folder)
            if m:
                info["date"] = f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"

        # HHMMSS, HHMM, or data_HHMMSS / data_HHMM
        time_str = re.sub(r"^data_", "", time_folder)  # strip optional data_ prefix
        time_str = time_str.split("_")[0]               # strip trailing _n suffix
        m = re.match(r"^(\d{2})(\d{2})(\d{2})?$", time_str)
        if m:
            hh, mm = m.group(1), m.group(2)
            ss = m.group(3) or "00"
            info["time"] = f"{hh}:{mm}:{ss}"

        info["sample_name"] = sample_folder

    return info


# ─────────────────────────────────────────────────────────────────────────────
#  TDMS stage-position scanner
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_FILE = "stage_cache.txt"


def _stage_cache_path(meas_dir: Path) -> Path:
    return meas_dir / _CACHE_FILE


def _load_stage_cache(meas_dir: Path) -> dict | None:
    p = _stage_cache_path(meas_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Invalidate if TDMS is newer than cache
        tdms = meas_dir / "ForceCurve.tdms"
        if tdms.exists() and tdms.stat().st_mtime > p.stat().st_mtime:
            return None
        return data
    except Exception:
        return None


def _save_stage_cache(meas_dir: Path, data: dict) -> None:
    try:
        _stage_cache_path(meas_dir).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def scan_tdms_stage(meas_dir: Path) -> dict:
    """Scan all force-curve indices in ForceCurve.tdms and return stage stats.

    Reads channel 2 (Zs — stage voltage) for every force curve, computes:
        U_dist : largest (max-min) range across all curves   [V]
        U_max  : absolute maximum stage voltage              [V]
        U_min  : absolute minimum stage voltage              [V]

    Results are cached in stage_cache.txt next to the TDMS file.
    Returns empty dict if TDMS is missing or unreadable.
    """
    tdms_path = meas_dir / "ForceCurve.tdms"
    if not tdms_path.exists():
        return {}

    # Try cache first
    cached = _load_stage_cache(meas_dir)
    if cached:
        return cached

    try:
        from nptdms import TdmsFile
        import numpy as np

        tdms = TdmsFile.open(str(tdms_path))
        channels = tdms["Forcecurve"].channels()
        if len(channels) < 3:
            return {}

        zs_ch = channels[2]
        zs_all = np.array(zs_ch[:])   # read entire channel at once (fast)

        # Get n_samples from config if available, else guess from data length
        n_fc = None
        cfg_path = meas_dir / "config.txt"
        if cfg_path.exists():
            text = _read_text(cfg_path) or ""
            m = re.search(r"FCあたりのデータ取得点数:\s*([0-9]+)", text)
            if m:
                n_fc = int(m.group(1))

        if n_fc and n_fc > 0 and len(zs_all) >= n_fc:
            n_curves = len(zs_all) // n_fc
            # Reshape: each row is one force curve's Zs channel
            zs_matrix = zs_all[: n_curves * n_fc].reshape(n_curves, n_fc)
            curve_ranges = zs_matrix.max(axis=1) - zs_matrix.min(axis=1)
            u_dist = float(curve_ranges.max())
        else:
            # Fallback: treat whole channel as one block
            u_dist = float(zs_all.max() - zs_all.min())

        result = {
            "u_dist": round(u_dist, 4),
            "u_max":  round(float(zs_all.max()), 4),
            "u_min":  round(float(zs_all.min()), 4),
        }
        _save_stage_cache(meas_dir, result)
        return result

    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────


# Matches YYMMDD, YYYYMMDD date folders
_RE_DATE   = re.compile(r"^\d{6}$|^\d{8}$")
# Matches HHMM, HHMM_n, HHMMSS, HHMMSS_n time/measurement folders
_RE_MEAS   = re.compile(r"^\d{4}(?:_\S+)?$|^\d{6}(?:_\S+)?$|^data_\d{4,6}(?:_\S+)?$")


def _safe_iterdir(p: Path) -> list:
    """iterdir() with full exception suppression — never crashes on bad filenames."""
    try:
        return [c for c in p.iterdir() if c.is_dir()]
    except Exception:
        return []


_MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi", ".gif",
               ".mp3", ".wav", ".pdf", ".zip"}


def _has_real_data(meas_dir: Path, data_glob: str) -> bool:
    """Return True if meas_dir has real measurement data.
    Show if: config.txt exists OR a non-media data file exists.
    Skip only if: completely empty OR contains only media/hidden files.
    """
    if (meas_dir / "config.txt").exists():
        return True
    try:
        for f in meas_dir.iterdir():
            if not f.name.startswith(".") and f.suffix.lower() not in _MEDIA_EXTS:
                return True
    except Exception:
        pass
    return False


def _find_measurement_configs(mode_root: Path, data_glob: str) -> list:
    """
    Walk ROOT/MODE/DATEDIR/MEASDIR/ looking for measurement folders.
    Only descends into date-shaped and measurement-shaped directories.
    Skips folders that contain only media files (png, mp4, etc.) with no data.
    Returns a list of config.txt Paths (may not exist if only data file found).
    """
    seen: set = set()
    cfgs: list = []

    for date_dir in _safe_iterdir(mode_root):
        if not _RE_DATE.match(date_dir.name):
            continue

        for meas_dir in _safe_iterdir(date_dir):
            if not _RE_MEAS.match(meas_dir.name):
                continue
            if not _has_real_data(meas_dir, data_glob):
                continue  # skip dirs with only screenshots/videos

            cfg = meas_dir / "config.txt"
            if meas_dir not in seen:
                seen.add(meas_dir)
                cfgs.append(cfg)

    # Also catch data files missed by the walk above
    try:
        for data_file in sorted(mode_root.rglob(data_glob)):
            d = data_file.parent
            if d not in seen:
                seen.add(d)
                cfgs.append(d / "config.txt")
    except Exception:
        pass

    return sorted(cfgs, key=lambda p: str(p))


def discover_pf_datasets(root: Path) -> list[dict]:
    """Scan ROOT/PF/**/ for measurement folders, return list of dataset dicts."""
    pf_root = root / "PF"
    datasets = []
    if not pf_root.exists():
        return datasets

    for cfg in _find_measurement_configs(pf_root, "ForceCurve.tdms"):
        path_info = parse_pf_path(cfg)
        cfg_data  = parse_pf_config(cfg)
        comments  = load_comments(cfg.parent)

        stage = scan_tdms_stage(cfg.parent)
        datasets.append({
            "mode":        "PF",
            "config_path": str(cfg),
            "folder":      str(cfg.parent),
            "date":        path_info["date"],
            "time":        path_info["time"],
            "meas_id":     path_info["meas_id"],
            "cantilever":  comments.get("cantilever") or cfg_data.get("cantilever"),
            "frequency_hz": cfg_data["frequency_hz"],
            "n_samples":   cfg_data["n_samples"],
            "x_length":    cfg_data["x_length"],
            "y_length":    cfg_data["y_length"],
            "x_step":      cfg_data["x_step"],
            "y_step":      cfg_data["y_step"],
            "u_amplitude": cfg_data["u_amplitude"],
            "u_trigger":   cfg_data["u_trigger"],
            "phase_start": cfg_data["phase_start"],
            "phase_end":   cfg_data["phase_end"],
            "u_dist":      stage.get("u_dist"),
            "u_max":       stage.get("u_max"),
            "u_min":       stage.get("u_min"),
            "comments":    comments.get("comments", ""),
            "has_config":  _dataset_is_filled(cfg_data, comments),
        })

    return datasets


def _find_fv_configs(fv_root: Path) -> list:
    """
    Walk ROOT/FV/YYYYMMDD/SAMPLENAME/HHMMSS/ looking for measurement folders.
    SAMPLENAME can be any string — we don't filter it.
    HHMMSS folder must match the time pattern.
    """
    seen: set = set()
    cfgs: list = []

    for date_dir in _safe_iterdir(fv_root):
        if not _RE_DATE.match(date_dir.name):
            continue
        for sample_dir in _safe_iterdir(date_dir):
            # SAMPLENAME: skip hidden files/macOS junk but allow any real folder name
            if sample_dir.name.startswith("."):
                continue
            for time_dir in _safe_iterdir(sample_dir):
                if not _RE_MEAS.match(time_dir.name):
                    continue
                if not _has_real_data(time_dir, "ForceCurve_*.lvm"):
                    continue  # skip completely empty dirs
                cfg = time_dir / "config.txt"
                if time_dir not in seen:
                    seen.add(time_dir)
                    cfgs.append(cfg)

    return sorted(cfgs, key=lambda p: str(p))


def discover_fv_datasets(root: Path) -> list[dict]:
    """Scan ROOT/FV/**/ for measurement folders, return list of dataset dicts."""
    fv_root = root / "FV"
    datasets = []
    if not fv_root.exists():
        return datasets

    for cfg in _find_fv_configs(fv_root):
        path_info = parse_fv_path(cfg)
        cfg_data  = parse_fv_config(cfg)
        comments  = load_comments(cfg.parent)

        datasets.append({
            "mode":        "FV",
            "config_path": str(cfg),
            "folder":      str(cfg.parent),
            "date":        path_info["date"],
            "time":        path_info["time"],
            "sample_name": comments.get("sample_name") or path_info["sample_name"],
            "sample_name_original": path_info["sample_name"],
            "cantilever":  comments.get("cantilever") or cfg_data.get("cantilever"),
            "u_trigger":   cfg_data["u_trigger"],
            "app_speed":   cfg_data["app_speed"],
            "ret_speed":   cfg_data["ret_speed"],
            "x_length":    cfg_data["x_length"],
            "y_length":    cfg_data["y_length"],
            "x_step":      cfg_data["x_step"],
            "y_step":      cfg_data["y_step"],
            "num_approach": cfg_data["num_approach"],
            "num_retract":  cfg_data["num_retract"],
            "loop_time":    cfg_data["loop_time"],
            "ret_length":   cfg_data["ret_length"],
            "filter":       cfg_data["filter"],
            "velocity_app": cfg_data["velocity_app"],
            "velocity_ret": cfg_data["velocity_ret"],
            "comments":    comments.get("comments", ""),
            "has_config":  _dataset_is_filled(cfg_data, comments),
        })

    return datasets


def discover_all(root: Path) -> list[dict]:
    """Return all PF + FV datasets sorted newest-first."""
    pf = discover_pf_datasets(root)
    fv = discover_fv_datasets(root)
    all_ds = pf + fv

    def sort_key(d):
        return (d.get("date") or "0000-00-00", d.get("time") or "00:00:00")

    return sorted(all_ds, key=sort_key, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Edit operations
# ─────────────────────────────────────────────────────────────────────────────

def update_dataset_meta(folder: str, mode: str, updates: dict) -> dict:
    """
    Persist editable metadata for a dataset.
    - comments, cantilever → stored in .afm_comments.json sidecar
    - sample_name (FV only) → also renames the parent folder on disk
    - creates a NaN stub config.txt if none exists yet
    Returns {"ok": True, "new_folder": str, "now_filled": bool}.
    """
    folder_path = Path(folder)
    config_path = folder_path / "config.txt"

    # Create stub config if missing
    _create_nan_config(config_path, mode)

    comments = load_comments(folder_path)

    if "comments" in updates:
        comments["comments"] = updates["comments"]
    if "cantilever" in updates:
        comments["cantilever"] = updates["cantilever"]

    new_folder = folder_path  # may be reassigned below

    if mode == "FV" and "sample_name" in updates:
        new_name = updates["sample_name"].strip()
        if new_name:
            # FV layout: .../YYMMDD/SAMPLENAME/HHMMSS/
            # folder_path = HHMMSS dir, its parent = SAMPLENAME dir
            sample_dir = folder_path.parent       # SAMPLENAME folder
            new_sample_dir = sample_dir.parent / new_name

            # Only rename if name actually changed
            if new_sample_dir != sample_dir:
                try:
                    sample_dir.rename(new_sample_dir)
                    new_folder = new_sample_dir / folder_path.name
                except Exception as e:
                    return {"ok": False, "error": f"Could not rename folder: {e}"}

            comments["sample_name"] = new_name

    try:
        save_comments(new_folder, comments)
    except Exception as e:
        return {"ok": False, "error": f"Could not save comments: {e}"}

    # Check if the dataset now has real values (for badge update)
    cfg_data = parse_pf_config(new_folder / "config.txt") if mode == "PF" else parse_fv_config(new_folder / "config.txt")
    now_filled = _dataset_is_filled(cfg_data, comments)

    return {"ok": True, "new_folder": str(new_folder), "now_filled": now_filled}


# ─────────────────────────────────────────────────────────────────────────────
#  LVM I/O (Force Volume)
# ─────────────────────────────────────────────────────────────────────────────

def load_lvm(filepath: Path):
    """Read LVM file → (Z_V, D_V) arrays (unsplit)."""
    data = np.loadtxt(filepath)
    if data.ndim != 1:
        raise ValueError(f"{filepath.name}: expected 1-D data, got {data.shape}")
    n = len(data) // 2
    if n == 0:
        raise ValueError(f"{filepath.name}: file too short")
    return data[n:2 * n], data[:n]   # Z, D


def split_turnaround(Z: np.ndarray, D: np.ndarray, t: int | None = None):
    if t is None:
        t = int(np.argmax(Z))
    t = int(np.clip(t, 1, len(Z) - 2))
    return Z[:t + 1], D[:t + 1], Z[t:], D[t:]


def list_force_curves(directory: Path) -> list[Path]:
    def _idx(p):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    return sorted(directory.glob("ForceCurve_*.lvm"), key=_idx)
