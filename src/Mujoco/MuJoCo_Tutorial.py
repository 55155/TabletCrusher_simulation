import mujoco
import numpy as np
import mediapy as media

path = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_V3_Pjoint_description/urdf/Crank_slider_system_V3_Pjoint_sensor.xml"

model = mujoco.MjModel.from_xml_path(path)

try:
    model.geom()
except KeyError as e:
    print(e)
for i in range(model.nbody):
    print(model.body(i).name)

data = mujoco.MjData(model)

m, d = model, data
print(data.geom_xpos)
mujoco.mj_kinematics(m, d)
print(data.time)
print(data.geom_xpos)
duration = 7
framerate = 60
frames = []

# enable joint visualization option: 
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if len(frames) < int(data.time * framerate):
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
            mujoco.viewer.launch(model)