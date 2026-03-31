import genesis as gs

gs.init(backend=gs.cpu, logging_level="debug")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        # run_in_thread=True,
    ),
    show_viewer=True,
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
scene.add_entity(
    gs.morphs.URDF(
        file='/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/TEST_LINK_sizeup_description/urdf/TEST_LINK_sizeup.urdf',
        scale=0.001,
    ),
)
scene.build()

for i in range(100):
    scene.step()
