import mujoco
import numpy as np
path = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_V3_Pjoint_description/urdf/Crank_slider_system_V3_Pjoint.xml"
model = mujoco.MjModel.from_xml_string(path)
sim = mujoco.MjModel.MjSim(model)
viewer = mujoco.MjModel.MjViewer(sim)

# 관절 개수에 맞게 위치 배열 가져오기
qpos = sim.data.qpos

# 원하는 관절 인덱스에 원하는 위치 할당 (예: 첫 번째 관절을 0.5 라디안으로 설정)
qpos[0] = 0.5

# 변경된 위치를 시뮬레이션에 반영
sim.data.qpos[:] = qpos

# 시뮬레이션 스텝 및 렌더링
for _ in range(1000):
    sim.step()
    viewer.render()
