"""
make_demo_data.py — Generate sample AFM directory trees for local testing.

Usage:
    python make_demo_data.py [root_dir]

Default root: ~/AFM_data   (used by server.py on Mac/Linux)
On Windows, pass D:\\シュテファン as the argument.
"""

import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "AFM_data"


PF_CONFIG_TEMPLATE = """\
FCあたりのデータ取得点数: {n_samples}
XStep:  {x_step}
YStep:  {y_step}
X計測範囲(μm):  {x_size:.6f}
Y計測範囲(μm):  {y_size:.6f}
周波数(Hz):  {freq:.6f}
データ取得開始位相:  0.000000
データ取得終了位相:  0.990000
ZSample gain(P, I, D):  0.000000, 0.000000, 0.000000
振幅(V):  {u_amp:.6f}
トリガ電圧(V):  {u_trig:.6f}
空振り時PID係数:  3.000000
CP閾値Deflection(V):  0.100000
ZSample gain(P, I, D):  0.050000, 0.061351, 0.000000
カンチレバー種類:  {cantilever}
"""

FV_CONFIG_TEMPLATE = """\
start_time,{start_time}
end_time,{end_time}
Vtrig,{vtrig:.6f}
Zig,FALSE
num_app,{num_app}
num_ret,{num_ret}
xlength,{xlength:.6f}
ylength,{ylength:.6f}
filter,no filter
app_speed_ratio,{app_speed:.6f}
ret_speed_ratio,{ret_speed:.6f}
Xstep,{x_step}
Ystep,{y_step}
Xstep_count,0
Ystep_count,0
loop_time,{loop_time:.6f}
FIFO_loop,20.000000
ret_length,8.000000
sr,FALSE
sr_time,1.000000
sr_loop,50000.000000
move_distance,config.txt
default_x(um)0.000000
default_y(um)0.000000
"""

SAMPLES  = ["Bone_Dry", "Bone_PBS", "Cartilage_01", "Collagen_Gel", "ControlGlass"]
CANT     = ["AC40", "AC160", "AC240"]
FREQS    = [50, 100, 200, 500, 1000]

def hhmmss(dt):
    return dt.strftime("%H%M%S")

def yymmdd(dt):
    return dt.strftime("%y%m%d")

def rand_dt(base, offset_min):
    return base + timedelta(minutes=offset_min)

def write_pf(root, dt, meas_id):
    date_dir = root / "PF" / yymmdd(dt)
    meas_dir = date_dir / f"{hhmmss(dt)}_{meas_id}"
    meas_dir.mkdir(parents=True, exist_ok=True)
    cfg = PF_CONFIG_TEMPLATE.format(
        n_samples=random.choice([250, 500, 1000]),
        x_step=random.choice([5, 10, 20]),
        y_step=random.choice([5, 10, 20]),
        x_size=random.uniform(0, 30),
        y_size=random.uniform(0, 30),
        freq=random.choice(FREQS),
        u_amp=random.uniform(0.3, 1.0),
        u_trig=random.uniform(0.1, 0.5),
        cantilever=random.choice(CANT),
    )
    (meas_dir / "config.txt").write_text(cfg, encoding="utf-8")
    (meas_dir / "ForceCurve.tdms").write_bytes(b"fake tdms data")  # marker file
    print(f"  PF  {meas_dir.relative_to(root)}")

def write_fv(root, dt, sample_name):
    date_dir   = root / "FV" / yymmdd(dt)
    sample_dir = date_dir / sample_name
    time_dir   = sample_dir / hhmmss(dt)
    time_dir.mkdir(parents=True, exist_ok=True)
    t0 = dt.hour * 10000 + dt.minute * 100 + dt.second
    cfg = FV_CONFIG_TEMPLATE.format(
        start_time=f"{t0:.2f}",
        end_time=f"{t0 + random.randint(60, 600):.2f}",
        vtrig=random.uniform(0.05, 0.3),
        num_app=random.choice([60000, 120000, 240000]),
        num_ret=random.choice([60000, 120000, 240000]),
        xlength=random.uniform(0, 20),
        ylength=random.uniform(0, 20),
        app_speed=random.uniform(1.0, 3.0),
        ret_speed=random.uniform(1.0, 3.0),
        x_step=random.choice([2, 5, 10]),
        y_step=random.choice([2, 5, 10]),
        loop_time=random.choice([10, 20, 50]),
    )
    (time_dir / "config.txt").write_text(cfg, encoding="utf-8")
    print(f"  FV  {time_dir.relative_to(root)}")


if __name__ == "__main__":
    print(f"Generating demo data in: {ROOT}\n")
    random.seed(42)

    # Generate ~3 days of mixed measurements
    base = datetime(2025, 3, 14, 9, 0, 0)
    offset = 0
    for day in range(3):
        day_base = base + timedelta(days=day)
        offset = 0
        n = random.randint(4, 8)
        for _ in range(n):
            offset += random.randint(10, 45)
            dt = rand_dt(day_base, offset)
            if random.random() < 0.5:
                write_pf(ROOT, dt, random.choice(["bone", "ctrl", "1", "2", "ref"]))
            else:
                write_fv(ROOT, dt, random.choice(SAMPLES))

    print(f"\nDone. Now run: uvicorn server:app --reload --port 8000")
    print(f"Then open: http://localhost:8000")
