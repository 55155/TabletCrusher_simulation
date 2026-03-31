import numpy as np
import genesis as gs
import yaml
import argparse
# /home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet/Tablet_description.xml
# "/Users/csrc_autonomouslab/Desktop/Seongjin/genesis_simulator_linux/src/config.yaml"
# delete equality tag in xml to see the effect of equality constraint
import os
pre_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("-dt", "--timestep", type=float, default=0.01, help="Simulation time step")
parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization GUI")
parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("-t", "--seconds", type=float, default=2.0, help="Number of seconds to simulate")
parser.add_argument("-f", "--force", action="store_true", default=True, help="Use ContactForceSensor (xyz float)")
parser.add_argument("-nf", "--no-force", action="store_false", dest="force", help="Use ContactSensor (boolean)")

args = parser.parse_args()
try:
    with open(pre_path+"/src/config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
except FileNotFoundError as e:
    print(e)

gs.init(backend=gs.metal if args.cpu else gs.gpu, logging_level=None)

scene = gs.Scene(
    show_viewer=True,
    sim_options= gs.options.SimOptions(
        dt = 0.01,
        gravity=(0.0, 0.0, -9.81),
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    # renderer=gs.renderers.RayTracer(),
    renderer=gs.renderers.Rasterizer(),
)
plane = scene.add_entity(gs.morphs.Plane(
    pos = (0, 0, 0),
))
cam = scene.add_camera(
    res=(1280, 960),
    pos=(2.0 * np.sin(1 / 60), 2.0 * np.cos(np.pi), 1),
    lookat=(0, 0, 0.0),
    fov=30,
    
    GUI=True,
)
solver = scene.sim.rigid_solver
# Adding a drone entity to the scene
fn= pre_path + '/My_asset/Tablet/Tablet_description.xml'

tablet = scene.add_entity(
    gs.morphs.MJCF( file = fn,
                    scale = 10.0, 
                    pos = (0,0,10), 
                    euler = (0,0,0), 
                    decimate = False, 
                    convexify = False,),
)
body_name = ["Tablet", "segment"]
link_idx = [tablet.get_link(name).idx_local for name in body_name]
link1 = tablet.get_link(body_name[0])
link2 = tablet.get_link(body_name[1])
link1_idx_arr = np.array(link1.idx, dtype=gs.np_int)
link2_idx_arr = np.array(link2.idx, dtype=gs.np_int)

scene.build()

solver.add_weld_constraint(link1_idx_arr, link2_idx_arr)
for i in range(1000):
    scene.step()
    if i == 500:
        print("Resetting tablet position and clearing equalities", tablet.equalities)
        print("After clearing equalities", tablet.equalities)
        solver.delete_weld_constraint(link1_idx_arr, link2_idx_arr)
        state = scene.get_state()
        scene.reset(state)
        print("After clearing equalities", tablet.equalities)

