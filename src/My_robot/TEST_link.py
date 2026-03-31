'''
import genesis as gs

gs.init(backend=gs.cuda)

Scene = gs.Scene(show_viewer=True,
                 sim_options= gs.options.SimOptions(
                    dt = 0.01,
                    gravity=(0.0, 0.0, -9.81),
                    ),
                )

Scene.add_entity(gs.morphs.Plane())
Scene.add_entity(gs.morphs.URDF(
    file=config["file_name"],                
    decimate=True,
    scale = 1.0,)
)

cam = Scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    
    GUI=True,
)

Scene.build()
cam.start_recording()

for i in range(1000):
    Scene.step()
    cam.render()

cam.stop_recording() # save_to_filename=f"/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/video/{config['file_name'].split('/')[-1]}_NO_decimation.mp4", fps=60)
'''
import genesis as gs

gs.init(backend=gs.cpu, logging_level="debug")

scene = gs.Scene(
    show_viewer=True,
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
scene.add_entity(
    gs.morphs.Mesh(
        file='/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/TEST_LINK_sizeup_description/meshes/base_link.stl',
        scale=(0.001, 0.001, 0.001),
    ),
)
scene.build()
# scene.viewer.run()