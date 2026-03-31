import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer=True,
    sim_options= gs.options.SimOptions(
        dt = 0.01,
        gravity=(0.0, 0.0, 0.0),
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
    pos = (0, 0, -1),
))

# Adding a drone entity to the scene
# franka = scene.add_entity(
#     gs.morphs.URDF(file = 'urdf/drones/racer.urdf'),
# )
file_name = 'xml/franka_emika_panda/panda.xml'

# Adding a My_link entity to the scene
my_link = scene.add_entity(
    gs.morphs.MJCF(file = file_name,),
)

cam = scene.add_camera(
    res=(640, 480),
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

cam.stop_recording(save_to_filename=f"/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/video/{file_name.split('/')[-1]}.mp4", fps=60)