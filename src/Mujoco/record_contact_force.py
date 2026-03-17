"""
Contact Force Recorder  —  viewer + video recording
  · gear ratio 선택 가능 (--gear)
  · --material 로 재질 선택 (steel / al)
  · --torque 로 crank 출력 토크(N·m) 직접 지정 (ctrl = torque / gear)
  · --ctrl 로 raw control input 직접 지정 (--torque 와 동시 사용 불가)
  · contact force, crank RPM 기록
  · CSV + PNG + MP4 저장

Usage:
    python record_contact_force.py --gear 229 --material al --torque 2.0 --steps 10000
    python record_contact_force.py --gear 229 --torque 2.5 --steps 10000
    python record_contact_force.py --gear 229 --ctrl 0.001 --steps 1000 --no-video
"""

import argparse
import os
import csv
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio.v3 as iio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False

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
parser.add_argument("--no-video", action="store_true",
                    help="동영상 저장 비활성화 (기본: 저장)")
parser.add_argument("--video-fps", type=int, default=60,
                    help="저장할 MP4 FPS (default: 60)")
parser.add_argument("--video-width",  type=int, default=1280,
                    help="동영상 가로 해상도 (default: 1280)")
parser.add_argument("--video-height", type=int, default=720,
                    help="동영상 세로 해상도 (default: 720)")
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
    """Impact_plate <-> tablet 법선(normal) 방향 접촉력 반환. 없으면 0.0

    mj_contactForce 는 contact-frame 기준 6D wrench 를 반환한다:
      ft_buf[0]  : normal force  (법선 방향, 항상 양수)
      ft_buf[1~2]: tangential force (마찰)
      ft_buf[3~5]: moment
    법선 분력만 사용하므로 월드 프레임 변환 불필요.
    """
    F_normal = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        b1  = model.geom_bodyid[con.geom1]
        b2  = model.geom_bodyid[con.geom2]
        if not ((b1 == bid_plate and b2 == bid_tablet) or
                (b1 == bid_tablet and b2 == bid_plate)):
            continue
        mujoco.mj_contactForce(model, data, i, ft_buf)
        F_normal += abs(ft_buf[0])   # normal force only
    return F_normal if np.isfinite(F_normal) else 0.0


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

# ── 비디오 렌더러 초기화 ──────────────────────────────────────────────────
frames = []
if not args.no_video:
    if not IMAGEIO_OK:
        print("[video] imageio 미설치 → 비디오 저장 비활성화. "
              "pip install \"imageio[ffmpeg]\" 로 설치하세요.")
        args.no_video = True
    else:
        renderer = mujoco.Renderer(model,
                                   height=args.video_height,
                                   width=args.video_width)
        # 캡처 간격: 시뮬 timestep 기준으로 목표 FPS에 맞게 자동 계산
        capture_every = max(1, int(round(1.0 / (args.video_fps * model.opt.timestep))))
        print(f"[video] {args.video_width}x{args.video_height}  "
              f"{args.video_fps} fps  capture_every={capture_every} steps")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 0.6
    for step in range(args.steps):
        data.ctrl[act_id] = args.ctrl
        mujoco.mj_step(model, data)

        omega = data.qvel[dof_adr]          # rad/s (crank joint)

        steps_arr[step]  = step
        times_arr[step]  = data.time
        forces_arr[step] = contact_force_mag(ft_buf)
        rpm_arr[step]    = omega * 60.0 / (2.0 * np.pi)

        viewer.sync()

        # 프레임 캡처
        if not args.no_video and step % capture_every == 0:
            renderer.update_scene(data, camera=-1)   # -1: free camera
            frames.append(renderer.render())
# viewer 블록 종료 → 뷰어 자동 닫힘

if not args.no_video:
    renderer.close()

# ── 결과 출력 ─────────────────────────────────────────────────────────────
print("\nDone. ── Contact Force (normal) Summary ──────────────────────────")
print(f"  peak  Fn          = {forces_arr.max():.4f} N")
if (forces_arr > 0).any():
    contact_mask = forces_arr > 0
    print(f"  mean  Fn (contact only)  = {forces_arr[contact_mask].mean():.4f} N")
    print(f"  min   Fn (contact only)  = {forces_arr[contact_mask].min():.4f} N")
    print(f"  std   Fn (contact only)  = {forces_arr[contact_mask].std():.4f} N")
    print(f"  contact steps / total    = {contact_mask.sum()} / {args.steps}")
else:
    print("  no contact detected")
print("──────────────────────────────────────────────────────────────────")

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

# ── MP4 저장 ───────────────────────────────────────────────────────────────
if not args.no_video and frames:
    mp4_path = os.path.join(OUT_DIR, f"sim_{label}_{args.steps}steps.mp4")
    try:
        iio.imwrite(mp4_path, np.stack(frames), fps=args.video_fps, codec="libx264")
        print(f"MP4  saved -> {mp4_path}  ({len(frames)} frames)")
    except Exception as e:
        print(f"[video] MP4 저장 실패: {e}")
        print("[video] pip install \"imageio[ffmpeg]\" 확인 후 재시도하세요.")

# ── PNG 저장 ───────────────────────────────────────────────────────────────
plt.rcParams["font.family"]       = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(13, 5))

color_f = "tab:blue"
color_r = "tab:orange"

ax1.plot(steps_arr, forces_arr, color=color_f, linewidth=0.8, label="Fn (normal)")
ax1.fill_between(steps_arr, forces_arr, alpha=0.15, color=color_f)
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.set_xlabel("Step")
ax1.set_ylabel("Fn normal [N]", color=color_f)
ax1.tick_params(axis="y", labelcolor=color_f)

ax2 = ax1.twinx()
ax2.plot(steps_arr, rpm_arr, color=color_r, linewidth=0.7, alpha=0.8, label="Crank RPM")
ax2.set_ylabel("Crank RPM", color=color_r)
ax2.tick_params(axis="y", labelcolor=color_r)

ax1.set_title(
    f"Contact Force & Crank RPM  |  gear={args.gear}  material={args.material}  "
    f"{torque_label}  steps={args.steps}\n"
    f"peak Fn={forces_arr.max():.1f} N  |  "
    f"mean Fn(contact)={forces_arr[forces_arr>0].mean():.1f} N  |  "
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
