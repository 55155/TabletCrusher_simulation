import argparse

import numpy as np

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-n", "--n_envs", type=int, default=0)
    parser.add_argument("-f", "--force", action="store_true", default=True, help="Use ContactForceSensor (xyz float)")

    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
        res=(1080, 960),
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )
    cam = scene.add_camera(
        res=(1280, 960),
        pos = (2,-2,2),
        lookat=(0.65, 0.0, 0.036),
        fov=30,
        GUI=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    
    bottle_link = [
        'base',
        # 'link_0',
        # # 'link_1',
        # 'link_0_helper'
        
    ]
    print("bottle_link_name : ", bottle)
    # Tablet link
    bottle_links = [bottle.get_link(name) for name in bottle_link]
    print("bottle_links : ", bottle_links)
    """
        'link_1' : [globla_idx, local_idx]
        'link_2' : [globla_idx, local_idx]
        ...
    """
    bottle_idx = {bottle_link[i]: [None, None] for i in range(len(bottle_link))}
    # na전역 0, 지역 1
    for i, name in enumerate(bottle_link):
        bottle_idx[name][0] = bottle_links[i].idx
        bottle_idx[name][1] = bottle_links[i].idx_local

    # add sensors to the scene
    for link_name in bottle_link:
        if args.force:
            sensor_options = gs.sensors.ContactForce(
                entity_idx=bottle.idx,
                link_idx_local=bottle.get_link(link_name).idx_local,
                draw_debug=True,
            )
            plot_kwargs = dict(
                title=f"{link_name} Force Sensor Data",
                labels=["force_x", "force_y", "force_z"],
                window_size=(1000, 900),
            )
        else:
            sensor_options = gs.sensors.Contact(
                entity_idx=bottle.idx,
                link_idx_local=bottle.get_link(link_name).idx_local,
                draw_debug=True,
            )
            plot_kwargs = dict(
                title=f"{link_name} Contact Sensor Data",
                labels=["in_contact"],
            )

        sensor = scene.add_sensor(sensor_options)

        if IS_PYQTGRAPH_AVAILABLE:
            sensor.start_recording(gs.recorders.PyQtLinePlot(**plot_kwargs))
        elif IS_MATPLOTLIB_AVAILABLE:
            print("pyqtgraph not found, falling back to matplotlib.")
            sensor.start_recording(gs.recorders.MPLLinePlot(**plot_kwargs))
        else:
            print("matplotlib or pyqtgraph not found, skipping real-time plotting.")
    
    fingers = [
        "left_finger",
        "right_finger",
    ]

    for finger in fingers: 
        sensor_options = gs.sensors.ContactForce(
            entity_idx=franka.idx,
            link_idx_local=franka.get_link(finger).idx_local,
            draw_debug=True,
        )
        plot_kwargs = dict(
            title=f"{finger} Force Sensor Data",
            labels=["force_x", "force_y", "force_z"],
            window_size=(1000, 900),
    )
    sensor = scene.add_sensor(sensor_options)
    sensor.start_recording(gs.recorders.PyQtLinePlot(**plot_kwargs))


    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))
    cam.start_recording()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    if args.n_envs == 0:
        franka.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
    else:
        franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.25]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    qpos[..., -2:] = 0.04

    path = franka.plan_path(qpos)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()
        cam.render()
    for i in range(30):
        scene.step()
        cam.render()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.142]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.142]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    for i in range(100):
        scene.step()
        cam.render()

    # grasp
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    for i in range(100):
        scene.step()
        cam.render()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.3]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-20, -20]) if args.n_envs == 0 else np.array([[-20, -20]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(1000):
        pre_force = franka.get_dofs_force(fingers_dof)
        print("finger force: ", pre_force)
        cam.set_pose(
            pos = (1.15, -0.5, 0.3),
            lookat=(0.65, 0.0, 0.3),
        )
        scene.step()
        cam.render()

    cam.stop_recording(save_to_filename ="video/grasp_bottle2.mp4")

    


if __name__ == "__main__":
    main()
