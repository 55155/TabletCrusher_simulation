import mujoco
model = mujoco.MjModel.from_xml_path("Crank_slider_system_copy_modified.urdf")
mujoco.mj_saveLastXML("Crank_slider_system_copy_modified_inertia.xml", model)

