import time

import mujoco
import mujoco.viewer
path = "/Users/csrc_autonomouslab/Desktop/Seongjin/genesis_simulator_linux/My_asset/Crank_slider_system_V3_Pjoint_description/urdf/Crank_slider_system_V3_Pjoint_sensor.xml"
# path = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_V3_Pjoint_description/urdf/Crank_slider_system_V3_Pjoint_sensor.xml"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)

d.ctrl[0] = 0.6

# viewer option setting
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  # Run the viewer until it is closed or 30 seconds have elapsed.
  # 만약 뷰어가 유효하다면, 30초 동안 실행
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    # True False 반복, 2초마다 바뀜
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

    # print(mujoco.mj_fwdActuation(m, d))
  
    # print(mujoco.mj_getTotalmass(m))  # 모델의 총 질량 출력