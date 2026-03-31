
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
    pos=(0.5, 0.5, 0.5),
    lookat=(0.05, 0.05, 0.1), 
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
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0, 1)))
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0.4, 1)))
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0, 0)))
Samplebag.fix_particles(Samplebag.find_closest_particle((0.1, 0.4, 0)))

cam.start_recording()

# a = Samplebag.find_closest_particle((-1, 1, 1.0))
# b = Samplebag.find_closest_particle((1, 1, 1.0)) 

# print("좌측상단 : ", a)
# print("우측상단 : ", b)
# print("중간상단 :", (a+b)/2)

# particle_pos = Samplebag.get_particles_pos()
# left_top_pos = particle_pos[a]
# right_top_pos = particle_pos[b]
# center_top_pos = (left_top_pos + right_top_pos) / 2
# print("중앙상단 좌표 :", center_top_pos)

# # tablet.set_pos(center_top_pos)


horizon = 1000
for i in range(horizon):
    print(f"Step {i+1}/{horizon}")
    scene.step()
    print("Scene stepped.")
    # if i == 100:
    #     tablet.set_pos((0.0500, 0.0500, 0.0500))
    cam.render()
cam.stop_recording(save_to_filename="./video/ipc_solver_samplebag/[20260120] Samplebag(PBD)-tablet(Rigid) Default_coupler.mp4")
