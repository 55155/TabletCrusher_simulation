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
        gravity=(0.0, 0.0, -20),
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
# franka = scene.add_entity(
#     gs.morphs.URDF(file = 'urdf/drones/racer.urdf'),
# )
fn = '/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_copy_description/urdf/1.urdf'

# Adding a My_link entity to the scene
my_link = scene.add_entity(
    gs.morphs.URDF(file = fn, euler = (90,0,0),pos = (-0.3, 0.0, 0), scale = 1.0, decimate = False, convexify = False,),
)


## scene에 모든 엔티티 추가 후에 build
cam = scene.add_camera(
    res=(1280, 960),
    pos=(2.0 * np.sin(1 / 60), 2.0 * np.cos(np.pi), 1),
    lookat=(0, 0, 0.0),
    fov=30,
    
    GUI=True,
)

n_envs = 2
scene.build(n_envs=n_envs, env_spacing=(0.5, 0.5),)

jnt_names = [
    'link1_link2_joint',
    'link2_link3_joint',
    'link3_link4_joint',
    'link4_link5_joint',
]
dofs_idx = [my_link.get_joint(name).dof_idx_local for name in jnt_names]
joint_value = [0 for _ in range(len(dofs_idx))]

# for parallelization
pos_command = np.array(joint_value)[None, :].repeat(n_envs, axis=0)

# cam.start_recording()
# normal = cam.render()
iter = 1000

# my_link.control_dofs_position(pos_command, dofs_idx)

for i in range(iter):
    scene.step()
#     cam.set_pose(
#         pos=(2.0 * np.sin(1 / 60), 2.0 * np.cos(np.pi), i/20),
#         lookat=(i/20, i/20, 0.0),
#     )
#     cam.render()


# cam.stop_recording(save_to_filename = config['file_path']['video']+'/'+fn.split('/')[-1]+"_GF_env_300_rotatePI_black_high_res_no_devimate.mp4")