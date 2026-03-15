"""
Contact Force Recorder  —  1000 steps headless simulation
  · gear ratio 선택 가능 (--gear)
  · contact force 없으면 0 으로 기록
  · CSV + PNG 저장

Usage:
    python record_contact_force.py --gear 229
    python record_contact_force.py --gear 1 --ctrl 1.0 --steps 1000
"""

import argparse
import os
import csv
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# ── argument ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gear",  type=float, required=True,
                    help="Gear ratio (0.01 / 1 / 229)")
parser.add_argument("--ctrl",  type=float, default=0.001,
                    help="Control input (default: 0.001)")
parser.add_argument("--steps", type=int,   default=1000,
                    help="Number of simulation steps (default: 1000)")
args = parser.parse_args()

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../My_asset/Scene_description"))
OUT_DIR   = os.path.normpath(os.path.join(BASE_DIR, "../../results"))
os.makedirs(OUT_DIR, exist_ok=True)

GEAR_MAP = {
    0.01 : "Scene.xml",
    1.0  : "Scene_gear1.xml",
    229.0: "Scene_gear229.xml",
}

if args.gear not in GEAR_MAP:
    available = ", ".join(str(k) for k in GEAR_MAP)
    raise ValueError(f"gear={args.gear} not defined. Available: {available}")

scene_file = GEAR_MAP[args.gear]
label      = f"gear{args.gear}".replace(".", "p")

# ── 모델 로드 ──────────────────────────────────────────────────────────────
os.chdir(SCENE_DIR)
model = mujoco.MjModel.from_xml_path(scene_file)
data  = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

act_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")
bid_plate  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft_1")
bid_tablet = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablet")

if act_id == -1:
    raise RuntimeError("Actuator 'crank_motor' not found.")


def contact_force_mag(ft_buf: np.ndarray) -> float:
    """Impact_plate <-> tablet 접촉력 합력 크기 반환. 없으면 0.0"""
    F_mag = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        b1  = model.geom_bodyid[con.geom1]
        b2  = model.geom_bodyid[con.geom2]
        if not ((b1 == bid_plate and b2 == bid_tablet) or
                (b1 == bid_tablet and b2 == bid_plate)):
            continue
        mujoco.mj_contactForce(model, data, i, ft_buf)
        R        = con.frame.reshape(3, 3)
        f_world  = R.T @ ft_buf[:3]
        F_impact = f_world if b2 == bid_plate else -f_world
        F_mag   += np.linalg.norm(F_impact)
    return F_mag if np.isfinite(F_mag) else 0.0


# ── 시뮬레이션 ────────────────────────────────────────────────────────────
print(f"Running {args.steps} steps  (gear={args.gear}, ctrl={args.ctrl}) ...")

steps_arr  = np.zeros(args.steps, dtype=int)
times_arr  = np.zeros(args.steps)
forces_arr = np.zeros(args.steps)
ft_buf     = np.zeros(6)

for step in range(args.steps):
    data.ctrl[act_id] = args.ctrl
    mujoco.mj_step(model, data)

    steps_arr[step]  = step
    times_arr[step]  = data.time
    forces_arr[step] = contact_force_mag(ft_buf)

print("Done.")
print(f"  peak |F| = {forces_arr.max():.4f} N")
print(f"  mean |F| (contact only) = "
      f"{forces_arr[forces_arr > 0].mean():.4f} N" if (forces_arr > 0).any() else "  no contact detected")

# ── CSV 저장 ───────────────────────────────────────────────────────────────
os.chdir(BASE_DIR)
csv_path = os.path.join(OUT_DIR, f"contact_force_{label}_{args.steps}steps.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "time_s", "contact_force_N"])
    for s, t, fv in zip(steps_arr, times_arr, forces_arr):
        writer.writerow([s, f"{t:.6f}", f"{fv:.6f}"])

print(f"CSV  saved -> {csv_path}")

# ── PNG 저장 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps_arr, forces_arr, color="tab:blue", linewidth=0.8, label="|F|")
ax.fill_between(steps_arr, forces_arr, alpha=0.15, color="tab:blue")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_title(
    f"Contact Force  |  gear={args.gear}  ctrl={args.ctrl}  steps={args.steps}\n"
    f"peak={forces_arr.max():.3f} N  |  "
    f"timestep={model.opt.timestep:.4f} s  |  "
    f"total_time={times_arr[-1]:.3f} s",
    fontsize=10,
)
ax.set_xlabel("Step")
ax.set_ylabel("|F| [N]")
ax.grid(True, alpha=0.4)
ax.legend(fontsize=9)
plt.tight_layout()

png_path = os.path.join(OUT_DIR, f"contact_force_{label}_{args.steps}steps.png")
fig.savefig(png_path, dpi=150)
print(f"PNG  saved -> {png_path}")
plt.show()
