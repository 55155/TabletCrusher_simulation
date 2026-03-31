import time
import csv
import mujoco
import mujoco.viewer

SCENE_XML = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Scene_description/Scene.xml"
OUT_CSV   = "tablet_force_log.csv"

ACT_NAME   = "crank_motor"
CTRL_VALUE = 10.0

LOG_EVERY   = 1
MAX_SAMPLES = 6000

TABLET_BODY = "tablet"


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    crank_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACT_NAME)
    if crank_act_id < 0:
        raise ValueError(f"Actuator not found: {ACT_NAME}")

    tablet_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TABLET_BODY)
    if tablet_bid < 0:
        raise ValueError(f"Body not found: {TABLET_BODY}")

    rows = []
    step_count = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # 시작부터 계속 토크/ctrl 입력 유지
                data.ctrl[crank_act_id] = CTRL_VALUE

                mujoco.mj_step(model, data)

                # cfrc_ext 계산(모델에 force/acc 센서가 없으면 직접 호출이 필요할 수 있음)[web:61]
                mujoco.mj_rnePostConstraint(model, data)

                if step_count % LOG_EVERY == 0:
                    wrench = data.cfrc_ext[tablet_bid].copy()
                    f = wrench[3:6]  # 6D = [rot(3), tran(3)]에서 force 성분[web:52]
                    rows.append([float(data.time), float(f[0]), float(f[1]), float(f[2])])

                    # 6000개 샘플이면 종료
                    if len(rows) >= MAX_SAMPLES:
                        break

                viewer.sync()
                time.sleep(model.opt.timestep)
                step_count += 1

    finally:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "Fx_N", "Fy_N", "Fz_N"])
            writer.writerows(rows)

        print(f"Saved {len(rows)} samples to: {OUT_CSV}")


if __name__ == "__main__":
    main()
