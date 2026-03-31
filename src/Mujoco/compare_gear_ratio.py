"""
Gear Ratio Comparison  —  Contact Force / RPM & Crank Torque
  · 동일한 ctrl input으로 여러 gear ratio 결과를 한 화면에 비교
  · Subplot 1 : Time [s]  vs  Contact Force normal [N]  +  Crank RPM (우축)
  · Subplot 2 : Time [s]  vs  Crank Net Torque [N·m]
                  = qfrc_actuator  +  qfrc_constraint  +  qfrc_passive
                  (접촉 반력이 크랭크 DOF에 투영된 값을 포함)

[토크 계산 참고]
  MuJoCo 운동방정식:  M·q̈ = qfrc_actuator + qfrc_constraint + qfrc_passive + qfrc_applied − qfrc_bias
  · qfrc_actuator  = ctrl × gear  → ctrl 고정이면 상수  (이전 버전 문제)
  · qfrc_constraint = 접촉/구속 반력이 일반화좌표로 투영된 값
  · qfrc_passive    = 댐핑 등 수동력
  → 세 항의 합이 크랭크에 실제로 작용하는 순 구동력

Usage:
    python compare_gear_ratio.py --gears 1 229 --ctrl 0.001 --steps 5000
    python compare_gear_ratio.py --gears 0.01 1 229 --ctrl 0.005 --steps 10000 --material steel
"""

import argparse
import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# ── argument ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gears",    type=float, nargs="+", required=True,
                    help="비교할 gear ratio 목록 (예: 1 229)")
parser.add_argument("--material", type=str, default="steel",
                    choices=["steel", "al"],
                    help="재질 선택: steel (기본) / al (알루미늄)")
parser.add_argument("--ctrl",     type=float, default=0.001,
                    help="Raw control input (모든 gear에 동일 적용, default: 0.001)")
parser.add_argument("--steps",    type=int,   default=5000,
                    help="Number of simulation steps (default: 5000)")
args = parser.parse_args()

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../My_asset/Scene_description"))
OUT_DIR   = os.path.normpath(os.path.join(BASE_DIR, "../../results"))
os.makedirs(OUT_DIR, exist_ok=True)

GEAR_MAP = {
    0.01 : {"steel": "Scene.xml"},
    1.0  : {"steel": "Scene_gear1.xml"},
    229.0: {"steel": "Scene_gear229.xml",
             "al"  : "Scene_gear229_al.xml"},
}

# ── gear ratio 유효성 검사 ─────────────────────────────────────────────────
for g in args.gears:
    if g not in GEAR_MAP:
        raise ValueError(f"gear={g} not defined. Available: {list(GEAR_MAP.keys())}")
    if args.material not in GEAR_MAP[g]:
        raise ValueError(f"material='{args.material}' not available for gear={g}. "
                         f"Available: {list(GEAR_MAP[g].keys())}")

# ── 헬퍼 ──────────────────────────────────────────────────────────────────
def contact_force_mag(model, data, bid_plate, bid_tablet, ft_buf):
    """shaft_1 <-> tablet 법선(normal) 접촉력 합산"""
    F_normal = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        b1  = model.geom_bodyid[con.geom1]
        b2  = model.geom_bodyid[con.geom2]
        if not ((b1 == bid_plate and b2 == bid_tablet) or
                (b1 == bid_tablet and b2 == bid_plate)):
            continue
        mujoco.mj_contactForce(model, data, i, ft_buf)
        F_normal += abs(ft_buf[0])
    return F_normal if np.isfinite(F_normal) else 0.0


def run_simulation(gear, material, ctrl, steps):
    """단일 gear ratio 시뮬레이션 실행 → dict 반환"""
    scene_file = GEAR_MAP[gear][material]

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
    jnt_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Revolute 10")
    dof_adr    = model.jnt_dofadr[jnt_id]

    if act_id == -1:
        raise RuntimeError("Actuator 'crank_motor' not found.")

    times_arr      = np.zeros(steps)
    forces_arr     = np.zeros(steps)
    rpm_arr        = np.zeros(steps)
    tau_act_arr    = np.zeros(steps)   # qfrc_actuator  (= ctrl × gear, 상수)
    tau_con_arr    = np.zeros(steps)   # qfrc_constraint (접촉 반력 투영)
    tau_pas_arr    = np.zeros(steps)   # qfrc_passive    (댐핑 등)
    ft_buf         = np.zeros(6)

    print(f"  gear={gear:>6}  ctrl={ctrl}  steps={steps} ...", end="", flush=True)
    for step in range(steps):
        data.ctrl[act_id] = ctrl
        mujoco.mj_step(model, data)

        omega = data.qvel[dof_adr]   # rad/s

        times_arr[step]   = data.time
        forces_arr[step]  = contact_force_mag(model, data, bid_plate, bid_tablet, ft_buf)
        rpm_arr[step]     = omega * 60.0 / (2.0 * np.pi)
        tau_act_arr[step] = data.qfrc_actuator[dof_adr]
        tau_con_arr[step] = data.qfrc_constraint[dof_adr]
        tau_pas_arr[step] = data.qfrc_passive[dof_adr]

    tau_net = tau_act_arr + tau_con_arr + tau_pas_arr

    peak_f = forces_arr.max()
    mean_f = forces_arr[forces_arr > 0].mean() if (forces_arr > 0).any() else 0.0
    print(f"  peak Fn={peak_f:.2f} N  mean Fn={mean_f:.2f} N  "
          f"tau_net peak={tau_net.max():.4f} N·m")

    os.chdir(BASE_DIR)
    return {
        "time"    : times_arr,
        "force"   : forces_arr,
        "rpm"     : rpm_arr,
        "tau_act" : tau_act_arr,
        "tau_con" : tau_con_arr,
        "tau_pas" : tau_pas_arr,
        "tau_net" : tau_net,
    }


# ── 시뮬레이션 실행 ────────────────────────────────────────────────────────
print(f"\nRunning comparison  ctrl={args.ctrl}  steps={args.steps}  material={args.material}")
print("─" * 60)

results = {}
for gear in args.gears:
    results[gear] = run_simulation(gear, args.material, args.ctrl, args.steps)

print("─" * 60)

# ── 플롯 ──────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False   # 음수 기호 깨짐 방지
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)

ax1_rpm = ax1.twinx()   # subplot 1 우측 Y축 (RPM)
ax2_con = ax2.twinx()   # subplot 2 우측 Y축 (qfrc_constraint)

for i, gear in enumerate(sorted(results.keys())):
    c   = colors[i % len(colors)]
    lbl = f"gear={gear}"
    r   = results[gear]

    # subplot 1 좌축: Contact Force
    ax1.plot(r["time"], r["force"], color=c, linewidth=0.9, label=f"{lbl}  Fn")

    # subplot 1 우측: RPM (점선)
    ax1_rpm.plot(r["time"], r["rpm"], color=c, linewidth=0.8,
                 linestyle="--", alpha=0.7, label=f"{lbl}  RPM")

    # subplot 2 좌축: tau_net (실선)
    ax2.plot(r["time"], r["tau_net"], color=c, linewidth=1.0,
             label=f"{lbl}  τ_net")

    # subplot 2 좌축: qfrc_passive (점점선 -.-)
    ax2.plot(r["time"], r["tau_pas"], color=c, linewidth=0.8,
             linestyle="-.", alpha=0.75, label=f"{lbl}  τ_passive")

    # subplot 2 우측: qfrc_constraint (점선)
    ax2_con.plot(r["time"], r["tau_con"], color=c, linewidth=0.8,
                 linestyle="--", alpha=0.8, label=f"{lbl}  τ_constraint")

# ── ax1 꾸미기 ────────────────────────────────────────────────────────────
ax1.set_ylabel("Contact Force  Fn [N]", fontsize=11)
ax1_rpm.set_ylabel("Crank RPM", fontsize=11, color="gray")
ax1_rpm.tick_params(axis="y", labelcolor="gray")

# 범례 통합 (좌 + 우축)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_rpm.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right",
           ncol=2)
ax1.grid(True, alpha=0.25)
ax1.set_title(
    f"Gear Ratio Comparison  |  ctrl={args.ctrl}  steps={args.steps}  material={args.material}",
    fontsize=11,
)

# ── ax2 꾸미기 ────────────────────────────────────────────────────────────
ax2.set_ylabel(r"Torque [N·m]  —  $\tau_{net}$(실선) / $\tau_{passive}$(점점선)",
               fontsize=10)
ax2_con.set_ylabel("τ_constraint  [N·m]  (점선)", fontsize=10, color="gray")
ax2_con.tick_params(axis="y", labelcolor="gray")
ax2.set_xlabel("Time  [s]", fontsize=11)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax2_con.axhline(0, color="gray", linewidth=0.3, linestyle=":")

# 범례 통합 (좌 + 우축)
lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_con.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, fontsize=8, loc="upper right", ncol=2)
ax2.grid(True, alpha=0.25)

plt.tight_layout()

gear_str = "_".join(str(g).replace(".", "p") for g in sorted(results.keys()))
png_path = os.path.join(OUT_DIR,
    f"compare_gear_{gear_str}_{args.material}_ctrl{str(args.ctrl).replace('.','p')}"
    f"_{args.steps}steps.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"\nPNG  saved -> {png_path}")
plt.show()
