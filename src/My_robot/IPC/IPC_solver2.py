
import argparse
import logging

import numpy as np
from huggingface_hub import snapshot_download
import genesis as gs

# gs.init()
# ## Basic 관련 설정
# scene = gs.Scene(
#     sim_options=gs.options.SimOptions(
#         dt       = 1e-4,
#         substeps = 40,
#     ),
#     viewer_options=gs.options.ViewerOptions(
#         camera_fov = 30,
#         res        = (1280, 720),
#         max_FPS    = 60,
#     ),
#     show_viewer = True,
# )
Samplebag_filename = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Sample_bag.obj"

## IPC 관련 설정

gs.init(backend=gs.cuda, logging_level=logging.INFO, performance_mode=True)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", default=True)
parser.add_argument("--vis_ipc", action="store_true", default=True)
args = parser.parse_args()

dt = 2e-3
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
    coupler_options=gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        contact_d_hat=0.01,  # Contact barrier distance (10mm) - must be appropriate for mesh resolution
        contact_friction_mu=0.3,  # Friction coefficient
        IPC_self_contact=False,  # Disable rigid self-contact in IPC
        two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
        disable_genesis_ground_contact=True,  # Disable Genesis ground contact to avoid double contact handling
        enable_ipc_gui=args.vis_ipc,
    ),
    show_viewer=args.vis,
)
args.vis = args.vis or args.vis_ipc
######################### materials #########################
rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
cloth_material = gs.materials.PBD.Cloth()
######################### entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=rigid_material,
)

cam = scene.add_camera(
    res=(1280, 960),
    pos=(0.5, 0.5, 0.5),
    lookat=(0.0500, 0.0500, 0.1000), 
) 

ball = scene.add_entity(
    material=rigid_material,
    morph=gs.morphs.Sphere(
        pos=(0.0500, 0.0500, 1.000),
        radius=0.500,
    ),
    surface=gs.surfaces.Default(
        smooth=True,
    ),
)

scene.build()
cam.start_recording()

horizon = 500
for i in range(horizon):
    print(f"Step {i+1}/{horizon}")
    scene.step()
    print("Scene stepped.")
    cam.render()

cam.stop_recording(save_to_filename="./video/ipc_solver_samplebag/[20260115] Samplebag(PBD)-tablet(Rigid).mp4")
