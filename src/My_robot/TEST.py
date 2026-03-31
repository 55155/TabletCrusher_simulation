import genesis
import numpy as np

genesis.init(backend=genesis.cuda)
scene = genesis.Scene(
    show_viewer=True,
    sim_options= genesis.options.SimOptions(
        dt = 0.01,
        gravity=(0.0, 0.0, -9.81),
    ),
    viewer_options=genesis.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=genesis.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    # renderer=genesis.renderers.RayTracer(),
    renderer=genesis.renderers.Rasterizer(),
)

plane = scene.add_entity(genesis.morphs.Plane(
    pos = (0, 0, 0),
))
# Adding a drone entity to the scene

test = scene.add_entity(
    genesis.morphs.MJCF(file = 'My_asset/TEST_description/urdf/TEST.xml',
                    scale = 5.0, pos = (0,0,1), euler = (0,0,0), decimate = False, convexify = False,),

)
scene.build()
jnt_name = [
    'Revolute 1'
]

dofs_idx = [test.get_joint(jnt).dof_idx_local for jnt in jnt_name]

print("DOF indices:", dofs_idx)
for i in range(1000):
    print(test.get_dofs_force())
    scene.step()

