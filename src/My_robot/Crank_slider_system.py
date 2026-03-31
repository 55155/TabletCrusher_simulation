import genesis as gs
import yaml
import numpy as np

with open('/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/src/config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.Loader)

gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer=True,
    sim_options= gs.options.SimOptions(
        dt = 0.01,
        gravity=(0.0, 0.0, 0),
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

# plane = scene.add_entity(gs.morphs.Plane(
#     pos = (0, 0, 0),
# ))

# Adding a drone entity to the scene
# franka = scene.add_entity(
#     gs.morphs.URDF(file = 'urdf/drones/racer.urdf'),
# )
fn = '/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_description/urdf/Crank_slider_system.urdf'

# Adding a My_link entity to the scene
my_link = scene.add_entity(
    gs.morphs.URDF(file = fn, euler = (90,0,0),),
)

cam = scene.add_camera(
    res=(1280, 960),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    
    GUI=True,
)

scene.build()

cam.start_recording()
normal = cam.render()

for i in range(300):
    scene.step()
    cam.render()
    cam.set_pose(
        pos=(-1.0 * np.sin(i / 120), 1.0 * np.cos(i / 120), 2),
        lookat=(0, 0, 0.0),
    )
    cam.render()

cam.stop_recording(save_to_filename = config['file_path']['video']+'/'+fn.split('/')[-1]+"_high_res_no_devimate.mp4")