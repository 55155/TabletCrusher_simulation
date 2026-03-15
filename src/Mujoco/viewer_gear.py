"""
MuJoCo Viewer  —  gear ratio selector + contact force direction reversal + real-time plot

Usage:
    python viewer_gear.py --gear 229
    python viewer_gear.py --gear 1
    python viewer_gear.py --gear 0.01
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── argument ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MuJoCo viewer with gear ratio selection")
parser.add_argument("--gear",      type=float, required=True,
                    help="Gear ratio (0.01 / 1 / 229)")
parser.add_argument("--ctrl",      type=float, default=0.001,
                    help="Control input magnitude (default: 0.001)")
parser.add_argument("--threshold", type=float, default=1.0,
                    help="Contact force threshold [N] for direction reversal (default: 1.0)")
parser.add_argument("--delay",     type=float, default=3.0,
                    help="Delay [s] before direction reversal after threshold (default: 3.0)")
parser.add_argument("--history",   type=int,   default=500,
                    help="Number of data points shown in plot (default: 500)")
args = parser.parse_args()

# ── Scene XML 매핑 ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../My_asset/Scene_description"))

GEAR_MAP = {
    0.01 : "Scene.xml",
    1.0  : "Scene_gear1.xml",
    229.0: "Scene_gear229.xml",
}

if args.gear not in GEAR_MAP:
    available = ", ".join(str(k) for k in GEAR_MAP)
    raise ValueError(f"gear={args.gear} is not defined. Available: {available}")

scene_file = GEAR_MAP[args.gear]
print(f"[viewer] gear={args.gear}  ctrl=+/-{args.ctrl}  "
      f"threshold={args.threshold} N  delay={args.delay} s  scene={scene_file}")

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


# ── 실시간 플롯 설정 ──────────────────────────────────────────────────────
N = args.history
buf_t = deque(maxlen=N)
buf_f = deque(maxlen=N)
reversal_times = []   # 방향 전환 발생 시각 (수직선 표시용)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle(f"Contact Force  |  gear={args.gear}  threshold={args.threshold} N  delay={args.delay} s",
             fontsize=11)
ax.set_xlabel("Time [s]")
ax.set_ylabel("|F| [N]")
ax.grid(True, alpha=0.4)

line_f,       = ax.plot([], [], color="tab:blue",   linewidth=1.0, label="|F|")
line_thresh   = ax.axhline(args.threshold, color="tab:red",    linewidth=0.8,
                            linestyle="--", label=f"threshold={args.threshold} N")
vlines = []   # 방향 전환 수직선

ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()

PLOT_EVERY = 5   # step마다 플롯 갱신 주기


def update_plot():
    if len(buf_t) < 2:
        return
    ts = list(buf_t)
    fs = list(buf_f)

    line_f.set_data(ts, fs)
    ax.set_xlim(ts[0], ts[-1])

    f_max = max(fs) if fs else 1.0
    ax.set_ylim(-0.05 * f_max, f_max * 1.15 + args.threshold * 0.1)

    # 새 방향 전환 수직선 추가
    while len(vlines) < len(reversal_times):
        vl = ax.axvline(reversal_times[len(vlines)], color="tab:orange",
                        linewidth=0.8, linestyle=":", alpha=0.7)
        vlines.append(vl)

    fig.canvas.draw_idle()
    plt.pause(0.001)


# ── 시뮬레이션 상태 ──────────────────────────────────────────────────────
ft_buf            = np.zeros(6)
direction         = 1
reversal_queue    = queue.Queue()
last_enqueue_time = -999.0
PRINT_EVERY       = 200

print("[viewer] Running - close the window to exit.")
print(f"         threshold={args.threshold} N  |  reversal delay={args.delay} s\n")

# ── 뷰어 실행 ────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 140
    viewer.cam.elevation = -20
    viewer.cam.distance  = 0.8

    step = 0
    while viewer.is_running():
        t_now = data.time

        # ── contact force 계산 ────────────────────────────────────────
        F_mag = contact_force_mag(ft_buf)
        buf_t.append(t_now)
        buf_f.append(F_mag)

        # ── threshold 초과 → queue에 예약 (delay * 2 이내 중복 무시) ────
        if F_mag >= args.threshold and (t_now - last_enqueue_time) > args.delay * 2:
            trigger_time = t_now + args.delay
            reversal_queue.put(trigger_time)
            last_enqueue_time = t_now
            print(f"[t={t_now:.3f}s]  |F|={F_mag:.4f} N >= {args.threshold} N  "
                  f"-> reversal queued at t={trigger_time:.3f}s")

        # ── queue에서 만료된 예약 실행 ────────────────────────────────
        while not reversal_queue.empty():
            trigger_time = reversal_queue.queue[0]
            if t_now >= trigger_time:
                reversal_queue.get()
                direction *= -1
                reversal_times.append(t_now)
                print(f"[t={t_now:.3f}s]  reversal executed  "
                      f"-> direction = {'+' if direction > 0 else '-'}")
            else:
                break

        # ── 제어 입력 적용 ────────────────────────────────────────────
        data.ctrl[act_id] = args.ctrl * direction

        mujoco.mj_step(model, data)
        viewer.sync()

        # ── 플롯 갱신 ─────────────────────────────────────────────────
        if step % PLOT_EVERY == 0:
            update_plot()

        if step % PRINT_EVERY == 0:
            print(f"[t={t_now:.3f}s]  |F|={F_mag:.4f} N  "
                  f"ctrl={data.ctrl[act_id]:.4f}  queued={reversal_queue.qsize()}")

        time.sleep(model.opt.timestep)
        step += 1

plt.ioff()
plt.show()
