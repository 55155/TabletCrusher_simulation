"""
claude_Crank-slider.py
======================
Crank actuator gear ratio 가 반력에 미치는 영향 비교.

케이스
  - Case A : gear = 1/229  (감속비 1/229)
  - Case B : gear = 1.0    (기어비 없음)

측정 대상
  1) tablet  — Impact plate 가 tablet 에 가하는 contact force
               (tablet joint 가 없는 fixed body → cfrc_ext 대신
                data.contact 에서 tablet geom 관련 force 를 직접 추출)
  2) shaft_1 — Impact plate 가 받는 constraint + contact force
               (cfrc_ext 로 정상 측정 가능)

각 케이스마다 MuJoCo viewer 를 열어 실시간으로 확인할 수 있다.
SIM_TIME 초 경과 후 viewer 가 자동으로 닫히고, 다음 케이스로 진행한다.
두 케이스 완료 후 비교 plot 을 gear_comparison.png 로 저장한다.

Usage (macOS):
    mjpython claude_Crank-slider.py

Usage (headless / Linux):
    python  claude_Crank-slider.py
"""

import time
import sys
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 저장 전용 — 뷰어와 분리
import matplotlib.pyplot as plt

# viewer 지원 여부 확인 (mjpython 필요)
try:
    import mujoco.viewer as _viewer_mod
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False

# ── 경로 설정 ────────────────────────────────────────────────────────────────
SCENE_XML  = ("/Users/csrc_autonomouslab/Desktop/Seongjin/"
              "genesis_simulation_on_linux/My_asset/"
              "Scene_description/Scene.xml")
OUTPUT_PNG = ("/Users/csrc_autonomouslab/Desktop/Seongjin/"
              "genesis_simulation_on_linux/src/Mujoco/"
              "gear_comparison.png")

# ── 시뮬레이션 파라미터 ──────────────────────────────────────────────────────
SIM_TIME   = 5.0    # 시뮬레이션 시간 [s]  (각 케이스마다 이 시간만큼 뷰어 표시)
CTRL_VAL   = 3.0    # actuator 제어 입력
VIEWER_DT  = 1/60   # 뷰어 갱신 주기 (≈ 60 fps)

CASES = {
    f"gear=1/229 ({1/229:.5f})  reduction": 1.0 / 229.0,
    "gear=1.0   no reduction":              1.0,
}


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def get_tablet_geom_ids(model: mujoco.MjModel) -> list[int]:
    tablet_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablet")
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == tablet_bid]


def contact_force_on_tablet(data: mujoco.MjData,
                             tablet_geoms: list[int]) -> np.ndarray:
    """data.contact 에서 tablet geom 과 접촉한 contact force 를 world frame 으로 합산."""
    f_total = np.zeros(3)
    for c in range(data.ncon):
        g1 = data.contact[c].geom1
        g2 = data.contact[c].geom2
        if g1 in tablet_geoms or g2 in tablet_geoms:
            cf = np.zeros(6)
            mujoco.mj_contactForce(data._model, data, c, cf)
            frame   = data.contact[c].frame.reshape(3, 3)
            f_total += frame.T @ cf[:3]
    return f_total


# ── 시뮬레이션 함수 ──────────────────────────────────────────────────────────
def run_simulation(gear_value: float, label: str) -> dict:
    """
    gear_value 로 시뮬레이션 실행.
    VIEWER_AVAILABLE = True  →  passive viewer 열어 실시간 표시
    VIEWER_AVAILABLE = False →  headless 실행
    """
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    model.actuator_gear[0, 0] = gear_value

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    shaft_bid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft_1")
    tablet_geoms = get_tablet_geom_ids(model)

    n_steps      = int(SIM_TIME / model.opt.timestep)
    step_per_frame = max(1, int(VIEWER_DT / model.opt.timestep))

    res = {k: np.empty(n_steps) for k in
           ("t",
            "tablet_fx", "tablet_fy", "tablet_fz",
            "shaft_fx",  "shaft_fy",  "shaft_fz")}
    total_contacts = 0

    def _step_and_record(i: int):
        nonlocal total_contacts
        data.ctrl[0] = CTRL_VAL
        mujoco.mj_step(model, data)
        mujoco.mj_rnePostConstraint(model, data)

        tf = contact_force_on_tablet(data, tablet_geoms)
        sw = data.cfrc_ext[shaft_bid]

        res["t"][i]          = data.time
        res["tablet_fx"][i]  = tf[0]; res["tablet_fy"][i] = tf[1]; res["tablet_fz"][i] = tf[2]
        res["shaft_fx"][i]   = sw[3]; res["shaft_fy"][i]  = sw[4]; res["shaft_fz"][i]  = sw[5]
        total_contacts += data.ncon

    # ── viewer 모드 ───────────────────────────────────────────────────────────
    if VIEWER_AVAILABLE:
        print(f"  [viewer] {label}  —  뷰어를 닫거나 {SIM_TIME:.0f}s 후 자동 종료")
        with _viewer_mod.launch_passive(model, data) as viewer:
            wall_start = time.time()
            i = 0
            while viewer.is_running() and i < n_steps:
                # step 묶음 실행
                for _ in range(step_per_frame):
                    if i >= n_steps:
                        break
                    _step_and_record(i)
                    i += 1

                viewer.sync()

                # 실시간 유지 (시뮬 시간 ≤ 실제 경과 시간)
                sim_elapsed  = i * model.opt.timestep
                wall_elapsed = time.time() - wall_start
                sleep_t      = sim_elapsed - wall_elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

                # 진행 상태 출력 (1초마다)
                if i % int(1.0 / model.opt.timestep) == 0:
                    s_mag = np.sqrt(res["shaft_fx"][i-1]**2
                                    + res["shaft_fy"][i-1]**2
                                    + res["shaft_fz"][i-1]**2)
                    t_mag = np.sqrt(res["tablet_fx"][i-1]**2
                                    + res["tablet_fy"][i-1]**2
                                    + res["tablet_fz"][i-1]**2)
                    print(f"    t={data.time:.1f}s  "
                          f"shaft|F|={s_mag:7.2f} N  "
                          f"tablet|F|={t_mag:7.2f} N  "
                          f"contacts={data.ncon}")

    # ── headless 모드 ─────────────────────────────────────────────────────────
    else:
        for i in range(n_steps):
            _step_and_record(i)

    res["total_contacts"] = total_contacts
    return res


# ── 실행 ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Gear Ratio Comparison  —  claude_Crank-slider.py")
print("=" * 60)

results: dict[str, dict] = {}
for label, gear in CASES.items():
    print(f"\n[CASE] {label}")
    r = run_simulation(gear, label)
    results[label] = r

    t_mag = np.sqrt(r["tablet_fx"]**2 + r["tablet_fy"]**2 + r["tablet_fz"]**2)
    s_mag = np.sqrt(r["shaft_fx"]**2  + r["shaft_fy"]**2  + r["shaft_fz"]**2)
    print(f"  steps={len(r['t'])}  total_contacts={r['total_contacts']}")
    print(f"  tablet |F|  max={t_mag.max():.4f} N   mean={t_mag.mean():.4f} N")
    print(f"  shaft  |F|  max={s_mag.max():.2f} N   mean={s_mag.mean():.2f} N")


# ── Plot 저장 ─────────────────────────────────────────────────────────────────
print("\n[PLOT] 비교 그래프 생성 중 ...")

palette = ["tab:blue", "tab:orange"]

fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
fig.suptitle(
    f"Reaction Force Comparison  |  ctrl = {CTRL_VAL}  |  {SIM_TIME} s\n"
    "Left: tablet  (contact force)     Right: shaft_1  (cfrc_ext)",
    fontsize=12,
)

comp_keys  = [("tablet_fx", "shaft_fx"),
              ("tablet_fy", "shaft_fy"),
              ("tablet_fz", "shaft_fz")]
comp_names = ["Fx [N]", "Fy [N]", "Fz [N]"]

for ax_pair, (tk, sk), cname in zip(axes[:3], comp_keys, comp_names):
    ax_t, ax_s = ax_pair
    for color, (label, r) in zip(palette, results.items()):
        ax_t.plot(r["t"], r[tk], color=color, linewidth=0.8, label=label)
        ax_s.plot(r["t"], r[sk], color=color, linewidth=0.8, label=label)
    for ax, title in ((ax_t, f"tablet  {cname}"), (ax_s, f"shaft_1  {cname}")):
        ax.set_ylabel(cname)
        ax.set_title(title, fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.legend(loc="upper right", fontsize=7)

for ax, bid_keys, title in zip(
        axes[3],
        [("tablet_fx","tablet_fy","tablet_fz"), ("shaft_fx","shaft_fy","shaft_fz")],
        ["tablet  |F| [N]", "shaft_1  |F| [N]"]):
    for color, (label, r) in zip(palette, results.items()):
        xk, yk, zk = bid_keys
        mag = np.sqrt(r[xk]**2 + r[yk]**2 + r[zk]**2)
        ax.plot(r["t"], mag, color=color, linewidth=0.9, label=label)
    ax.set_ylabel(title)
    ax.set_xlabel("time [s]")
    ax.set_title(title, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"[SAVED] {OUTPUT_PNG}")
print("완료.")
