
import argparse
import logging

import numpy as np
from huggingface_hub import snapshot_download
import genesis as gs

## INITIALIZE  GENESIS
gs.init(backend=gs.cuda, logging_level=logging.INFO, performance_mode=True)

# PARSER 설정
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", default=True)
parser.add_argument("--vis_ipc", action="store_true", default=True)
parser.add_argument("-n", "--n_envs", type=int, default=0)
parser.add_argument("-f", "--force", action="store_true", default=True, help="Use ContactForceSensor (xyz float)")

args = parser.parse_args()


# baseline 설정
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = 2e-3,
        substeps = 40,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov = 30,
        res        = (1280, 720),
        max_FPS    = 60,
    ),
    show_viewer = True,
)
Samplebag_filename = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Sample_bag.obj"

# # # IPC 관련 설정
# dt = 2e-3
# scene = gs.Scene(
#     sim_options=gs.options.SimOptions(dt=dt, substeps=20,gravity=(0.0, 0.0, -9.8)),
#     coupler_options=gs.options.IPCCouplerOptions(
#         dt=dt,
#         gravity=(0.0, 0.0, -9.8),
#         contact_d_hat=0.01,  # Contact barrier distance (10mm) - must be appropriate for mesh resolution
#         contact_friction_mu=0.3,  # Friction coefficient
#         IPC_self_contact=False,  # Disable rigid self-contact in IPC
#         two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
#         disable_genesis_ground_contact=True,  # Disable Genesis ground contact to avoid double contact handling
#         enable_ipc_gui=args.vis_ipc,
#     ),
#     show_viewer=args.vis,
# )
# args.vis = args.vis or args.vis_ipc
########################## materials ##########################
rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
cloth_material = gs.materials.PBD.Cloth()
######################### entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=rigid_material,
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos = (-.4,-.4,0)),
)
fingers = [
    "left_finger",
    "right_finger",
]
for finger in fingers: 
    sensor_options = gs.sensors.ContactForce(
        entity_idx=franka.idx,
        link_idx_local=franka.get_link(finger).idx_local,
        draw_debug=True,
    )
    plot_kwargs = dict(
        title=f"{finger} Force Sensor Data",
        labels=["force_x", "force_y", "force_z"],
        window_size=(1000, 900),
)
sensor = scene.add_sensor(sensor_options)
sensor.start_recording(gs.recorders.PyQtLinePlot(**plot_kwargs))



Samplebag = scene.add_entity(
    material=cloth_material,
    morph=gs.morphs.Mesh(
        pos = (0.0000, 0.0000, 0.0000),
        file = Samplebag_filename,
        scale=0.001,
        decimate=True,
        decimate_face_num=1000,
    ),
    surface=gs.surfaces.Default(
        smooth=True,
        color=(1.0, 1.0, 1.0, 1.0),
        opacity = 0.3,
        vis_mode='visual',
    )
)
cam = scene.add_camera(
    res=(1280, 960),
    pos=(2, 2, 2),
    lookat=(0., 0, 0), 
) 

# tablet = scene.add_entity(
#     material=rigid_material,
#     morph=gs.morphs.MJCF(
#         file = '/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet_posmod/Tablet_posmod.xml',
#         scale = 1.0, pos = (0,0,10), euler = (0,0,0), decimate = False, convexify = False,),
#     surface=gs.surfaces.Default(
#         smooth=False,
#     ),
#     vis_mode="collision",
#     visualize_contact=True,
# )

# tablet = scene.add_entity(
#     material=rigid_material,
#     morph = gs.morphs.Mesh(
#         file = '/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet_posmod/Tablet_posmod.obj',
#         scale = 0.002, pos = (0.0500, 0.0500, 1.1000), euler = (0,0,0),
#         decimate=True,          # 👈 추가
#         decimate_face_num=500,  # 👈 추가: 17k→500 faces
#         convexify=True,         # 👈 추가: 볼록 껍질로 collision 최적화
#     ),
#     vis_mode="collision",
#     visualize_contact=True,
# )
ball = scene.add_entity(
    material=rigid_material,
    morph=gs.morphs.Sphere(
        pos=(0.0500, 0.0500, 0.100),
        radius=0.0100,
    ),
    surface=gs.surfaces.Default(
        smooth=True,
    ),
)

scene.build()
cam.start_recording()
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# Optional: set control gains
if args.n_envs == 0:
    franka.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
else:
    franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)
end_effector = franka.get_link("hand")
# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.0500, 0.0500, .5]) if args.n_envs == 0 else np.array([[0.0500, 0.0500, .5]] * args.n_envs),
    quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs,),
)
qpos[..., -2:] = 0.04
qpos[6] = 2.8  # wrist joint adjustment

Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0, 1)))
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0.4, 1)))
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0, 0)))
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0.4, 0)))

path = franka.plan_path(qpos)
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    cam.render()
for i in range(30):
    scene.step()
    cam.render()

# reach
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.0500, 0.0500, 0.17]) if args.n_envs == 0 else np.array([[0.0500, 0.0500, 0.17]] * args.n_envs),
    quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs,),
)
qpos[..., -2:] = 0.1
qpos[6] = 2.8  # wrist joint adjustment

path = franka.plan_path(qpos)
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    cam.render()
for i in range(30):
    scene.step()
    cam.render()


# grasp
franka.control_dofs_position(qpos[..., :-2], motors_dof)
franka.control_dofs_position(
    np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
)  # you can use position control
for i in range(100):
    scene.step()
    cam.render()

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.05, 0.0, 1]) if args.n_envs == 0 else np.array([[0.05, 0.0, 1]] * args.n_envs),
    quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
)
franka.control_dofs_position(qpos[..., :-2], motors_dof)
franka.control_dofs_force(
    np.array([-20, -20]) if args.n_envs == 0 else np.array([[-200, -200]] * args.n_envs), fingers_dof
)  # can also use force control


horizon = 300
for i in range(horizon):
    print(f"Step {i+1}/{horizon}")
    if i == 0:
        Samplebag.release_particle(Samplebag.find_closest_particle((0.1, 0, 1)))
        Samplebag.release_particle(Samplebag.find_closest_particle((0.1, 0.4, 1)))
        Samplebag.release_particle(Samplebag.find_closest_particle((0.1, 0, 0)))
        Samplebag.release_particle(Samplebag.find_closest_particle((0.1, 0.4, 0)))
    scene.step()
    print("Scene stepped.")
    cam.render()
cam.stop_recording(save_to_filename="./video/ipc_solver_samplebag/[20260122] LagacyCoupler_Samplebag-Robotarm interaction 5.mp4")
