import numpy as np
import genesis as gs
import yaml

# with open('genesis_simulation_on_linux/src/config.yaml', 'r') as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)
# print(config)


########################## init ##########################
gs.init(backend=gs.cuda)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)


# --- (scene and robot creation omitted, identical to the sections above) ---
franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
    ),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.1, 0.1, 0.1),
        pos = np.array([0.65, 0.0, 0.28])
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    
    GUI=True,
)

scene.build()  # build the scene with a single environment
cam.start_recording()

# Retrieve some commonly used handles
rigid        = scene.sim.rigid_solver          # low-level rigid body solver
end_effector = franka.get_link("hand")        # Franka gripper frame
cube_link    = cube.get_link("box_baselink")   # the link we want to pick

################ Reach pre-grasp pose ################
q_pregrasp = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.13]),  # just above the cube
    quat = np.array([0, 1, 0, 0]),        # down-facing orientation
)
print(q_pregrasp)

franka.control_dofs_position(q_pregrasp[:-2], np.arange(7))  # arm joints only
for _ in range(50):
    cam.render()
    scene.step()

################ Attach (activate suction) ################
link_cube   = np.array([cube_link.idx],   dtype=gs.np_int)
link_franka = np.array([end_effector.idx], dtype=gs.np_int)
rigid.add_weld_constraint(link_cube, link_franka)

################ Lift and transport ################
q_lift = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.28]),  # lift up
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(q_lift[:-2], np.arange(7))
for _ in range(50):
    cam.render()
    scene.step()

q_place = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.4, 0.2, 0.18]),  # target place pose
    quat = np.array([0, 1, 0, 0]),
)

cam.start_recording()

franka.control_dofs_position(q_place[:-2], np.arange(7))
for _ in range(100):
    print(franka.get_dofs_force())
    cam.render()
    scene.step()

################ Detach (release suction) ################
rigid.delete_weld_constraint(link_cube, link_franka)
for _ in range(400):
    cam.render()
    scene.step()

print(config['video']['output_directory'])
cam.stop_recording(save_to_filename=f"{config['video']['output_directory']}/pick_and_place.mp4")  # save_to_filename="suctioncup_pick_and_place.mp4", fps=60)
