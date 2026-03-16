"""
Contact Force Recorder  —  headless simulation
  · gear ratio 선택 가능 (--gear)
  · --material 로 재질 선택 (steel / al)
  · --torque 로 crank 출력 토크(N·m) 직접 지정 (ctrl = torque / gear)
  · --ctrl 로 raw control input 직접 지정 (--torque 와 동시 사용 불가)
  · contact force, crank RPM 기록
  · CSV + PNG 저장

Usage:
    python record_contact_force.py --gear 229 --material al --torque 2.0 --steps 10000
    python record_contact_force.py --gear 229 --torque 2.5 --steps 10000
    python record_contact_force.py --gear 229 --ctrl 0.001 --steps 1000
"""

import argparse
import os
import csv
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# ── argument ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gear",     type=float, required=True,
                    help="Gear ratio (0.01 / 1 / 229)")
parser.add_argument("--material", type=str,   default="steel",
                    choices=["steel", "al"],
                    help="재질 선택: steel (기본) / al (알루미늄)")
parser.add_argument("--torque",   type=float, default=None,
                    help="Desired crank output torque [N·m] (ctrl = torque / gear)")
parser.add_argument("--ctrl",     type=float, default=None,
                    help="Raw control input (mutually exclusive with --torque)")
parser.add_argument("--steps",    type=int,   default=1000,
                    help="Number of simulation steps (default: 1000)")
args = parser.parse_args()

if args.torque is not None and args.ctrl is not None:
    parser.error("--torque and --ctrl are mutually exclusive.")
if args.torque is None and args.ctrl is None:
    args.ctrl = 0.001  # fallback default

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../My_asset/Scene_description"))
OUT_DIR   = os.path.normpath(os.path.join(BASE_DIR, "../../results"))
os.makedirs(OUT_DIR, exist_ok=True)

# gear × material → scene file 매핑
GEAR_MAP = {
    0.01 : {"steel": "Scene.xml"},
    1.0  : {"steel": "Scene_gear1.xml"},
    229.0: {"steel": "Scene_gear229.xml",
             "al"  : "Scene_gear229_al.xml"},
}

if args.gear not in GEAR_MAP:
    available = ", ".join(str(k) for k in GEAR_MAP)
    raise ValueError(f"gear={args.gear} not defined. Available: {available}")

mat_map = GEAR_MAP[args.gear]
if args.material not in mat_map:
    available_mat = ", ".join(mat_map.keys())
    raise ValueError(f"material='{args.material}' not available for gear={args.gear}. "
                     f"Available: {available_mat}")

scene_file = mat_map[args.material]
label      = f"gear{args.gear}_{args.material}".replace(".", "p")

# --torque → ctrl 자동 계산
if args.torque is not None:
    args.ctrl = args.torque / args.gear
    print(f"[torque mode] target torque = {args.torque} N·m  →  ctrl = {args.ctrl:.6f}")

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

# crank 관절 dof 주소 (RPM 계산용)
jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Revolute 10")
dof_adr = model.jnt_dofadr[jnt_id]


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
torque_label = (f"torque={args.torque} N·m" if args.torque is not None
                else f"ctrl={args.ctrl}")
print(f"Running {args.steps} steps  "
      f"(gear={args.gear}, material={args.material}, {torque_label}) ...")

steps_arr  = np.zeros(args.steps, dtype=int)
times_arr  = np.zeros(args.steps)
forces_arr = np.zeros(args.steps)
rpm_arr    = np.zeros(args.steps)
ft_buf     = np.zeros(6)

for step in range(args.steps):
    data.ctrl[act_id] = args.ctrl
    mujoco.mj_step(model, data)

    omega = data.qvel[dof_adr]          # rad/s (crank joint)

    steps_arr[step]  = step
    times_arr[step]  = data.time
    forces_arr[step] = contact_force_mag(ft_buf)
    rpm_arr[step]    = omega * 60.0 / (2.0 * np.pi)

# ── 결과 출력 ─────────────────────────────────────────────────────────────
print("Done.")
print(f"  peak  |F|       = {forces_arr.max():.4f} N")
if (forces_arr > 0).any():
    print(f"  mean  |F| (contact only) = {forces_arr[forces_arr > 0].mean():.4f} N")
else:
    print("  no contact detected")

# 후반부 1/4 구간 평균 RPM (정상상태 추정)
steady_start = int(args.steps * 0.75)
rpm_steady   = rpm_arr[steady_start:].mean()
rpm_all      = rpm_arr.mean()
print(f"  mean  RPM (all steps)     = {rpm_all:.2f} RPM")
print(f"  mean  RPM (steady, last 25%) = {rpm_steady:.2f} RPM")

# ── CSV 저장 ───────────────────────────────────────────────────────────────
os.chdir(BASE_DIR)
csv_path = os.path.join(OUT_DIR, f"contact_force_{label}_{args.steps}steps.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "time_s", "contact_force_N", "crank_rpm"])
    for s, t, fv, r in zip(steps_arr, times_arr, forces_arr, rpm_arr):
        writer.writerow([s, f"{t:.6f}", f"{fv:.6f}", f"{r:.4f}"])

print(f"CSV  saved -> {csv_path}")

# ── PNG 저장 ───────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(13, 5))

color_f = "tab:blue"
color_r = "tab:orange"

ax1.plot(steps_arr, forces_arr, color=color_f, linewidth=0.8, label="|F| contact")
ax1.fill_between(steps_arr, forces_arr, alpha=0.15, color=color_f)
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.set_xlabel("Step")
ax1.set_ylabel("|F| [N]", color=color_f)
ax1.tick_params(axis="y", labelcolor=color_f)

ax2 = ax1.twinx()
ax2.plot(steps_arr, rpm_arr, color=color_r, linewidth=0.7, alpha=0.8, label="Crank RPM")
ax2.set_ylabel("Crank RPM", color=color_r)
ax2.tick_params(axis="y", labelcolor=color_r)

ax1.set_title(
    f"Contact Force & Crank RPM  |  gear={args.gear}  material={args.material}  "
    f"{torque_label}  steps={args.steps}\n"
    f"peak F={forces_arr.max():.1f} N  |  "
    f"mean F(contact)={forces_arr[forces_arr>0].mean():.1f} N  |  "
    f"RPM_steady={rpm_steady:.1f}  |  "
    f"dt={model.opt.timestep:.4f} s  |  total={times_arr[-1]:.2f} s"
    if (forces_arr > 0).any() else
    f"Contact Force & Crank RPM  |  gear={args.gear}  material={args.material}  "
    f"{torque_label}  steps={args.steps}\n"
    f"no contact  |  RPM_steady={rpm_steady:.1f}  |  "
    f"dt={model.opt.timestep:.4f} s  |  total={times_arr[-1]:.2f} s",
    fontsize=9,
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
plt.tight_layout()

png_path = os.path.join(OUT_DIR, f"contact_force_{label}_{args.steps}steps.png")
fig.savefig(png_path, dpi=150)
print(f"PNG  saved -> {png_path}")
plt.show()
