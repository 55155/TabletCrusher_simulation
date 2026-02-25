import time
import csv
import mujoco
import mujoco.viewer
import numpy as np

SCENE_XML = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Scene_description/Scene.xml"

# 두 개 파일
IMPACT_CSV = "tablet_impact_plate_contact.csv"
WALL_CSV   = "tablet_wall1_contact.csv"

ACT_NAME   = "crank_motor"
CTRL_VALUE = 10.0

LOG_EVERY   = 1
MAX_SAMPLES = 6000
TABLET_BODY = "tablet"

IMPACT_KEYWORD = "Impact plate"  # geom 이름에 포함될 키워드
WALL_KEYWORD   = "Wall_1"


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

    # tablet geom ids
    tablet_geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == tablet_bid]

    # 두 CSV용 데이터
    impact_rows = []
    wall_rows = []

    step_count = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                data.ctrl[crank_act_id] = CTRL_VALUE

                mujoco.mj_step(model, data)

                if step_count % LOG_EVERY == 0:
                    t = float(data.time)

                    # Impact plate contact 찾기
                    impact_normal = 0.0
                    # Wall_1 contact 찾기
                    wall_normal = 0.0

                    for i in range(data.ncon):
                        con = data.contact[i]
                        
                        # Tablet 관련 contact만
                        if (con.geom1 in tablet_geom_ids) or (con.geom2 in tablet_geom_ids):
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(model, data, i, wrench)
                            
                            # Tablet이 받는 방향으로 부호 맞추기
                            normal = wrench[0] if con.geom1 in tablet_geom_ids else -wrench[0]
                            
                            # geom 이름으로 구분
                            g1_name = model.geom(gid=con.geom1).name if con.geom1>=0 else ""
                            g2_name = model.geom(gid=con.geom2).name if con.geom2>=0 else ""
                            pair_names = g1_name + g2_name
                            
                            if IMPACT_KEYWORD in pair_names:
                                impact_normal += normal
                            elif WALL_KEYWORD in pair_names:
                                wall_normal += normal

                    # 각 파일에 기록
                    impact_rows.append([t, impact_normal])
                    wall_rows.append([t, wall_normal])

                    # 6000개 샘플이면 종료
                    if len(impact_rows) >= MAX_SAMPLES:
                        break

                viewer.sync()
                time.sleep(model.opt.timestep)
                step_count += 1

    finally:
        # Impact plate CSV 저장
        with open(IMPACT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "normal_force_N"])
            writer.writerows(impact_rows)
        
        # Wall_1 CSV 저장
        with open(WALL_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "normal_force_N"])
            writer.writerows(wall_rows)

        print(f"Saved {len(impact_rows)} impact plate contacts to: {IMPACT_CSV}")
        print(f"Saved {len(wall_rows)} wall_1 contacts to: {WALL_CSV}")


if __name__ == "__main__":
    main()
