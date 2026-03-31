import time
import mujoco
import mujoco.viewer

import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="my_body" pos="0 0 1">
      <geom name="my_geom" type="sphere" size="0.1" rgba="1 0 0 1"/>
      <joint name="my_joint" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# XML 파일로부터 모델 로드
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Passive 뷰어로 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        
        # 물리 시뮬레이션 스텝
        mujoco.mj_step(model, data)
        
        # 뷰어 동기화
        viewer.sync()
        
        # 시간 유지
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
