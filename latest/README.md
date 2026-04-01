# AFM Logger

A fast, browser-based log viewer and annotation tool for Force Volume (FV)
and Pulse Force (PF) AFM measurements. Built with **FastAPI** + plain HTML/JS —
no Streamlit, no Electron, no npm. Runs on Windows and macOS.

---

## Directory layout expected

```
ROOT/
├── PF/
│   └── YYMMDD/
│       └── HHMMSS_n/        ← measurement folder (n = label/integer)
│           └── config.txt
└── FV/
    └── YYMMDD/
        └── SAMPLENAME/
            └── HHMMSS/
                └── config.txt
```

**Default ROOT:**
- Windows → `D:\シュテファン`
- Mac/Linux → `~/AFM_data`

Overridable at any time from the browser UI.

---

## Installation

```bash
pip install fastapi uvicorn[standard] nptdms numpy
```

Or:

```bash
pip install -r requirements.txt
```

---

## Running

**Windows:**
```
launch_windows.bat
```
Or manually:
```cmd
cd afm_logger
uvicorn server:app --reload --port 8000
```

**Mac / Linux:**
```bash
chmod +x launch_mac.sh
./launch_mac.sh
```

Then open **http://localhost:8000** in your browser.

---

## Demo data (testing without real measurements)

```bash
python make_demo_data.py          # creates ~/AFM_data with sample configs
# or on Windows:
python make_demo_data.py "D:\シュテファン"
```

---

## Features

### Display

| Column | PF | FV |
|--------|----|----|
| Mode badge | ✓ | ✓ |
| Time | ✓ | ✓ |
| Measurement ID / Sample Name | ID | Sample Name ✏️ |
| Cantilever | ✏️ | ✏️ |
| Frequency | ✓ | — |
| U_amplitude | ✓ | — |
| U_trigger | ✓ | ✓ |
| X × Y size | ✓ | ✓ |
| X × Y steps | ✓ | ✓ |
| n samples / speed | n / FC | n app / ret |
| Sampling phase range | ✓ | Loop time |
| Comments | ✏️ | ✏️ |

✏️ = inline editable — click to edit, Enter or ✓ to save.

### Editing

Click any editable cell to edit in place. Press **Enter** or the **✓ Save**
button to persist. Press **Escape** or **✗** to cancel.

Editable fields are stored in a `.afm_comments.json` sidecar file next to the
measurement's `config.txt` — the original config is **never modified**.

**Special:** editing `Sample Name` in an FV row will also **rename the
SAMPLENAME folder on disk** to match.

### Missing config

If no `config.txt` is found, all numeric fields show `—` and all fields
are treated as editable. A red `no cfg` badge appears on the row.

### Filtering & grouping

- All / PF / FV filter tabs at the top
- Grouped by date (newest on top)
- Within a day, consecutive measurements of the same mode share a table

---

## File overview

```
afm_logger/
├── server.py          ← FastAPI app (API routes)
├── afm_io.py          ← Unified I/O: config parsing, discovery, editing
├── make_demo_data.py  ← Test data generator
├── requirements.txt
├── launch_windows.bat
├── launch_mac.sh
└── static/
    └── index.html     ← Full frontend (HTML + CSS + JS, single file)
```

---

## Upcoming (planned)

- Per-row expandable panel with force curve preview and 2D maps
- Batch export of metadata as CSV/Excel
- Search / filter by sample name or date range
