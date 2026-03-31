import genesis as gs
import yaml
import numpy as np

# with open('/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/src/config.yaml', 'r') as file:
#     config = yaml.load(file, Loader=yaml.Loader)

gs.init()

scene = gs.Scene(
    show_viewer=True,
    sim_options= gs.options.SimOptions(
        dt = 0.0005,
        substeps = 10,
        gravity=(0.0, 0.0,0.0),
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=30,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    rigid_options=gs.options.RigidOptions(
        # requires_grad=True,
        # enable_collision=False, # TODO: remove this and you will see crazy behavior
    ),
    
    # renderer=gs.renderers.RayTracer(),
    renderer=gs.renderers.Rasterizer(),
)

# plane = scene.add_entity(gs.morphs.Plane(
#     pos = (0, 0, 0),
# ))
solver = scene.sim.rigid_solver

# fn = f'urdf/Crank_slider_system_V3_Pjoint_sensor_V2.xml'

Crank_slider_system = scene.add_entity(
        gs.morphs.MJCF(
            file = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crusher_description/urdf/Crusher.xml",
            pos = (0.0, 0.0,0.0),
            scale = 1.0,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )



## scene에 모든 엔티티 추가 후에 build
cam = scene.add_camera(
    res=(1280, 960),
    pos=(2.0 * np.sin(1 / 60), 2.0 * np.cos(np.pi), 1),
    lookat=(0, 0, 0.0),
    fov=30,
    
    GUI=True,
)

scene.build()

link_name = [
    "motor_shaft_1",
    # "Link2_1",
    # "Link3_1",
    # "Shaft_1",
]
links = [Crank_slider_system.get_link(name) for name in link_name]
link_idx = {link_name[i]: [None, None] for i in range(len(link_name))}

# na전역 0, 지역 1
for i, name in enumerate(link_name):
    link_idx[name][0] = links[i].idx
    link_idx[name][1] = links[i].idx_local


jnt_names = [
    'Revolute 10',
    # 'Revolute 12',
    # 'Revolute 13',
    # 'Slider 21',
]
dofs_idx = [Crank_slider_system.get_joint(name).dof_idx_local for name in jnt_names]

print(dofs_idx)

# parameters
r = 0.02 # crank radius
l = 0.08 # connecting rod length

# velocity command 
crank_velocity = 1/3* np.pi  # 1/3 pi rad/s
vel_command = np.array([crank_velocity,0,0])

# force command
crank_torque = 100.0  # N·m
epsilon = 1e-6
force_command = np.array([crank_torque])

cam.start_recording()
normal = cam.render()
iter = 1000

Crank_slider_system.set_dofs_kp(
    kp = np.array([0.1]),
    dofs_idx_local = dofs_idx,
)
Crank_slider_system.set_dofs_kv(
    kv = np.array([0.1]),
    dofs_idx_local = dofs_idx,
)


cam.set_pose(        
    pos = (10,2,5),
    lookat=(0, 2, 0.0),
)
# Crank_slider_system.control_dofs_force(force_command, dofs_idx)

# print(Crank_slider_system.get_dofs_force())

for i in range(iter):
    scene.step()
    Crank_slider_system.control_dofs_force(force_command, [0])

    # Crank_slider_system.control_dofs_position([0,0,0], dofs_idx)

    print(Crank_slider_system.get_dofs_force())

    cam.set_pose(        
        pos = (10,2,5),
        lookat=(0, 2, 0.0),
    )
    print(Crank_slider_system.get_dofs_force())
    cam.render()


# cam.stop_recording(save_to_filename =  "./video/Error_20251020_4.mp4")