import time
import mujoco
import mujoco.viewer
import numpy as np

SCENE_XML = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Scene_description/Scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

impact_plate_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft_1")
tablet_bid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablet")

# optional actuator
crank_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")

PRINT_EVERY = 10
step_count = 0
ft = np.zeros(6, dtype=np.float64)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if crank_act_id != -1:
            data.ctrl[crank_act_id] = 100.0

        mujoco.mj_step(model, data)

        if step_count % PRINT_EVERY == 0 and data.ncon > 0:
            # (옵션) 해당 스텝에서 Impact_plate에 작용한 접촉력 합력
            Fsum = np.zeros(3)

            for i in range(data.ncon):
                con = data.contact[i]
                g1, g2 = con.geom1, con.geom2
                b1 = model.geom_bodyid[g1]
                b2 = model.geom_bodyid[g2]

                # Impact_plate <-> tablet 만 선택
                is_pair = ((b1 == impact_plate_bid and b2 == tablet_bid) or
                           (b1 == tablet_bid and b2 == impact_plate_bid))
                if not is_pair:
                    continue

                # 6D force:torque in contact frame
                mujoco.mj_contactForce(model, data, i, ft)  # [web:11]
                f_c = ft[:3].copy()

                # contact frame -> world (simulate 코드도 동일한 방식으로 변환)  [web:19]
                R = con.frame.reshape(3, 3)
                f_world_on_geom2 = R.T @ f_c

                # mj_contactForce는 geom2에 작용하는 힘으로 보는 해석이 일반적이며,
                # geom1의 힘은 -1을 곱해 사용합니다. [web:14]
                if b2 == impact_plate_bid:
                    F_impact = f_world_on_geom2
                elif b1 == impact_plate_bid:
                    F_impact = -f_world_on_geom2
                else:
                    continue

                Fsum += F_impact

                # 더 작은 값도 보이게 소수점 자릿수 늘림
                print(f"[t={data.time:8.4f}] Impact_plate<->tablet  "
                      f"F_impact_world = [{F_impact[0]: .6f}, {F_impact[1]: .6f}, {F_impact[2]: .6f}]  "
                      f"|F|={np.linalg.norm(F_impact):.6f} N")

            # contact가 여러 개면 합력도 같이 보고 싶을 때
            if np.linalg.norm(Fsum) > 0:
                print(f"                SUM on Impact_plate = [{Fsum[0]: .6f}, {Fsum[1]: .6f}, {Fsum[2]: .6f}]  "
                      f"|SUM|={np.linalg.norm(Fsum):.6f} N")

        viewer.sync()
        time.sleep(model.opt.timestep)
        step_count += 1
