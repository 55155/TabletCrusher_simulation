import numpy as np
import genesis as gs
import yaml
# /home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet/Tablet_description.xml

with open('/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/src/config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.Loader)
gs.init(backend=gs.cuda)
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
# Adding a drone entity to the scene
fn='/home/seongjin/Desktop/Seongjin/' \
    'genesis_simulation_on_linux/My_asset/Tablet/Tablet_description.xml'

tablet = scene.add_entity(
    gs.morphs.MJCF(file = fn,
                    scale = 10.0, 
                    pos = (0,0,10), 
                    euler = (0,0,0), 
                    decimate = False, 
                    convexify = False,),
)
scene.build()

for i in range(1000):
    scene.step()