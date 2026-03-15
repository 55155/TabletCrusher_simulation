"""
Crank-Slider Contact Force Comparison  —  3 gear ratios
  · gear = 0.01  (Scene.xml,       original)
  · gear = 1     (Scene_gear1.xml, 1:1)
  · gear = 229   (Scene_gear229.xml, 1:229)

MuJoCo 는 MjModel / MjData 쌍을 여러 개 독립적으로 생성할 수 있으므로
세 환경을 동시에(같은 루프에서) step 하여 결과를 비교합니다.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Scene XML 경로 ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../My_asset/Scene_description"))

# MuJoCo C 백엔드는 한글 경로를 지원하지 않으므로
# SCENE_DIR 로 chdir 한 뒤 파일명만 넘깁니다.
CASES = {
    "original (gear=0.01)" : "Scene.xml",
    "gear 1:1  (gear=1)"   : "Scene_gear1.xml",
    "gear 1:229 (gear=229)" : "Scene_gear229.xml",
}
COLORS = {
    "original (gear=0.01)" : "tab:gray",
    "gear 1:1  (gear=1)"   : "tab:orange",
    "gear 1:229 (gear=229)" : "tab:blue",
}

# ── 시뮬레이션 파라미터 ────────────────────────────────────────────────────
SIM_DURATION = 3.0   # [s]
CTRL_INPUT   = 1.0   # 세 케이스 동일


# ── 헬퍼: body ID 기준 contact force 크기 합산 ────────────────────────────
def _contact_force_mag(model, data, bid_a, bid_b, ft_buf):
    """Impact_plate(bid_a) ↔ tablet(bid_b) 접촉력 합력 크기 반환."""
    F_mag = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        b1  = model.geom_bodyid[con.geom1]
        b2  = model.geom_bodyid[con.geom2]
        if not ((b1 == bid_a and b2 == bid_b) or
                (b1 == bid_b and b2 == bid_a)):
            continue
        mujoco.mj_contactForce(model, data, i, ft_buf)
        R        = con.frame.reshape(3, 3)
        f_world  = R.T @ ft_buf[:3]
        F_impact = f_world if b2 == bid_a else -f_world
        F_mag   += np.linalg.norm(F_impact)
    return F_mag


# ── 환경 초기화 ────────────────────────────────────────────────────────────
print("Loading environments ...")
os.chdir(SCENE_DIR)   # 한글 경로 우회: 상대경로로 XML 로드
envs = []
for label, xml_path in CASES.items():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")
    if act_id == -1:
        raise RuntimeError(f"[{label}] Actuator 'crank_motor' not found.")

    bid_plate  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft_1")
    bid_tablet = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablet")

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    n_steps = int(SIM_DURATION / model.opt.timestep)
    envs.append({
        "label"     : label,
        "model"     : model,
        "data"      : data,
        "act_id"    : act_id,
        "bid_plate" : bid_plate,
        "bid_tablet": bid_tablet,
        "n_steps"   : n_steps,
        "times"     : np.zeros(n_steps),
        "forces"    : np.zeros(n_steps),
        "ft_buf"    : np.zeros(6),
    })
    print(f"  OK  {label}")

# ── 동시 시뮬레이션 루프 ───────────────────────────────────────────────────
# MuJoCo 는 독립적인 (model, data) 쌍을 동시에 step할 수 있습니다.
n_steps_max = max(e["n_steps"] for e in envs)
print(f"\nRunning {n_steps_max} steps for all environments ...")

for step in range(n_steps_max):
    for e in envs:
        if step >= e["n_steps"]:
            continue
        e["data"].ctrl[e["act_id"]] = CTRL_INPUT
        mujoco.mj_step(e["model"], e["data"])
        e["times"][step]  = e["data"].time
        raw = _contact_force_mag(
            e["model"], e["data"],
            e["bid_plate"], e["bid_tablet"],
            e["ft_buf"]
        )
        # 수치 불안정(NaN/Inf) 방어
        e["forces"][step] = raw if np.isfinite(raw) else np.nan

print("Simulation complete.\n")

# ── 통계 출력 ──────────────────────────────────────────────────────────────
for e in envs:
    f = e["forces"]
    peak = f.max()
    mean = f[f > 0].mean() if (f > 0).any() else 0.0
    print(f"[{e['label']}]  peak = {peak:.4f} N  |  mean(contact) = {mean:.4f} N")

# ── 플롯: 개별 서브플롯 3개 ──────────────────────────────────────────────
fig, axes = plt.subplots(len(envs), 1, figsize=(12, 9), sharex=True)
fig.suptitle(
    f"Contact Force  —  3 Gear Ratios\n"
    f"ctrl = {CTRL_INPUT},  duration = {SIM_DURATION} s",
    fontsize=13,
)
for ax, e in zip(axes, envs):
    ax.plot(e["times"], e["forces"], color=COLORS[e["label"]], linewidth=0.9)
    f = e["forces"]
    peak = f.max()
    mean = f[f > 0].mean() if (f > 0).any() else 0.0
    ax.set_ylabel("|F| [N]")
    ax.set_title(f"{e['label']}   (peak={peak:.3f} N, mean={mean:.3f} N)")
    ax.grid(True, alpha=0.4)
axes[-1].set_xlabel("Time [s]")
plt.tight_layout()

# ── PNG 저장 ──────────────────────────────────────────────────────────────
OUT_DIR  = os.path.normpath(os.path.join(BASE_DIR, "../../results"))
os.chdir(BASE_DIR)   # 저장 경로 복원
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "contact_force_gear_comparison.png")
fig.savefig(OUT_PATH, dpi=150)
print(f"\nSaved → {OUT_PATH}")

plt.show()
