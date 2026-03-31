"""
MuJoCo Viewer  —  gear ratio selector + material + contact force direction reversal + real-time plot

Usage:
    python viewer_gear.py --gear 229 --material al --torque 2.5
    python viewer_gear.py --gear 229
    python viewer_gear.py --gear 229 --ctrl 0.001 --threshold 1.0 --delay 3.0
"""

import argparse
import os
import time
import queue
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np

# ── argument ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MuJoCo viewer with gear ratio selection")
parser.add_argument("--gear",      type=float, required=True,
                    help="Gear ratio (0.01 / 1 / 229)")
parser.add_argument("--material",  type=str,   default="steel",
                    choices=["steel", "al"],
                    help="재질 선택: steel (기본) / al (알루미늄)")
parser.add_argument("--torque",    type=float, default=None,
                    help="Crank output torque [N·m] (ctrl = torque / gear). --ctrl 과 상호 배타.")
parser.add_argument("--ctrl",      type=float, default=None,
                    help="Raw control input magnitude (default: 0.001)")
parser.add_argument("--window",    type=int,   default=50,
                    help="방향전환 판단용 angular velocity moving window 크기 (default: 50 steps)")
parser.add_argument("--zero_tol",  type=float, default=0.5,
                    help="방향전환 기준: window 평균 |omega| [rad/s] 이하이면 전환 (default: 0.5)")
parser.add_argument("--cooldown",  type=float, default=2.0,
                    help="방향전환 후 최소 대기 시간 [s] (default: 2.0)")
args = parser.parse_args()

if args.torque is not None and args.ctrl is not None:
    parser.error("--torque 와 --ctrl 은 동시에 사용할 수 없습니다.")
if args.torque is None and args.ctrl is None:
    args.ctrl = 0.001  # fallback default
if args.torque is not None:
    args.ctrl = args.torque / args.gear

# ── Scene XML 매핑 ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../My_asset/Scene_description"))

GEAR_MAP = {
    0.01 : {"steel": "Scene.xml"},
    1.0  : {"steel": "Scene_gear1.xml"},
    229.0: {"steel": "Scene_gear229.xml",
             "al"  : "Scene_gear229_al.xml"},
}

if args.gear not in GEAR_MAP:
    available = ", ".join(str(k) for k in GEAR_MAP)
    raise ValueError(f"gear={args.gear} is not defined. Available: {available}")

mat_map = GEAR_MAP[args.gear]
if args.material not in mat_map:
    raise ValueError(f"material='{args.material}' not available for gear={args.gear}. "
                     f"Available: {', '.join(mat_map.keys())}")

scene_file = mat_map[args.material]
torque_info = (f"torque={args.torque} N·m -> ctrl=+/-{args.ctrl:.6f}"
               if args.torque is not None else f"ctrl=+/-{args.ctrl}")
print(f"[viewer] gear={args.gear}  material={args.material}  {torque_info}  "
      f"window={args.window}  zero_tol={args.zero_tol} rad/s  cooldown={args.cooldown}s  "
      f"scene={scene_file}")

# ── 모델 로드 (한글 경로 우회) ─────────────────────────────────────────────
os.chdir(SCENE_DIR)
model = mujoco.MjModel.from_xml_path(scene_file)
data  = mujoco.MjData(model)

# ── 초기화 ────────────────────────────────────────────────────────────────
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")
if act_id == -1:
    raise RuntimeError("Actuator 'crank_motor' not found in model.")

bid_plate  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft_1")
bid_tablet = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablet")


def contact_force_mag(ft_buf: np.ndarray) -> float:
    """Impact_plate <-> tablet 접촉력 합력 크기 반환."""
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
    return F_mag


# ── crank 관절 dof 주소 ────────────────────────────────────────────────────
jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Revolute 10")
dof_adr = model.jnt_dofadr[jnt_id]

# ── 상태 변수 ─────────────────────────────────────────────────────────────
ft_buf         = np.zeros(6)
direction      = 1
peak_f         = 0.0
last_reversal  = -999.0          # 마지막 방향전환 시각 (cooldown 용)
omega_window   = deque(maxlen=args.window)   # 50-step moving window
PRINT_EVERY    = 200

print("[viewer] Running - close the window to exit.")
print(f"         window={args.window} steps  |  zero_tol={args.zero_tol} rad/s  "
      f"|  cooldown={args.cooldown}s\n")

# ── 뷰어 실행 ────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 140
    viewer.cam.elevation = -20
    viewer.cam.distance  = 0.8

    step = 0
    while viewer.is_running():
        t_now = data.time
        omega = data.qvel[dof_adr]          # rad/s
        rpm   = omega * 60.0 / (2.0 * np.pi)

        # ── contact force 계산 ────────────────────────────────────────
        F_mag = contact_force_mag(ft_buf)
        if F_mag > peak_f:
            peak_f = F_mag

        # ── 50-step moving window에 omega 추가 ────────────────────────
        omega_window.append(abs(omega))

        # ── 방향전환 판단 ─────────────────────────────────────────────
        # 조건: window가 가득 찼고, 평균 |omega| < zero_tol, cooldown 경과
        if (len(omega_window) == args.window
                and np.mean(omega_window) < args.zero_tol
                and (t_now - last_reversal) > args.cooldown):
            direction    *= -1
            last_reversal = t_now
            omega_window.clear()   # window 초기화 (즉시 재발동 방지)
            print(f"\n[t={t_now:.3f}s]  mean|ω|={np.mean(list(omega_window)+[abs(omega)]):.3f} rad/s "
                  f"< {args.zero_tol}  ->  reversal: {'CW' if direction > 0 else 'CCW'}")

        # ── 제어 입력 적용 ────────────────────────────────────────────
        data.ctrl[act_id] = args.ctrl * direction

        mujoco.mj_step(model, data)
        viewer.sync()

        # ── 콘솔 상태 출력 ────────────────────────────────────────────
        if step % PRINT_EVERY == 0:
            win_mean = np.mean(omega_window) if omega_window else 0.0
            bar_len  = min(int(F_mag / 10), 50)
            bar      = "█" * bar_len + "░" * (50 - bar_len)
            print(f"\r[t={t_now:6.2f}s]  RPM={rpm:5.1f}  |F|={F_mag:6.1f}N  "
                  f"peak={peak_f:6.1f}N  ω_avg={win_mean:.2f}  [{bar}]",
                  end="", flush=True)

        time.sleep(model.opt.timestep)
        step += 1

    print()  # 줄바꿈
