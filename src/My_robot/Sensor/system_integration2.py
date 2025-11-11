import argparse
import os
import numpy as np  
from tqdm import tqdm

import roma
import torch
import math as m

# 오일러 각을 회전 행렬로 변환
euler_angles = [90, 0, 90]  # degrees
R = roma.euler_to_rotmat('XYZ', euler_angles, degrees=True)

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=0.0005, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization GUI")
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=2.0, help="Number of seconds to simulate")
    parser.add_argument("-f", "--force", action="store_true", default=True, help="Use ContactForceSensor (xyz float)")
    parser.add_argument("-nf", "--no-force", action="store_false", dest="force", help="Use ContactSensor (boolean)")

    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cuda)
    # gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level=None)

    ########################## scene setup ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -9.81),
            dt=args.timestep,
        ),
        rigid_options=gs.options.RigidOptions(
            # constraint_timeconst -> weld 판단 
            # constraint_timeconst=max(0.01, 2 * args.timestep),
            use_gjk_collision=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=20,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )
    cam = scene.add_camera(
        res=(1280, 960),
        pos = (10,2,5),
        lookat=(0, 2, 0.0),
        fov=30,
        
        GUI=True,
    )

    # rigid solver : for add_constaraints
    solver = scene.sim.rigid_solver

    # Crank-slider system
    Crank_slider_system = scene.add_entity(
        gs.morphs.MJCF(
            file = "./My_asset/Crusher_description/urdf/" \
            "Crusher.xml",
            pos = (0, 0.0, 0),
            scale = 10.0,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
    )

    link_name = [
    "motor_shaft_1",
    "Link2_1",
    "Link3_1",
    "shaft_1",
    "Wall_1"
    ]
    links = [Crank_slider_system.get_link(name) for name in link_name]
    print("links : ", links)
    link_idx = {link_name[i]: [None, None] for i in range(len(link_name))}
    print("link_idx before : ", link_idx)
    # 전역 0, 지역 1
    for i, name in enumerate(link_name):
        link_idx[name][0] = links[i].idx
        link_idx[name][1] = links[i].idx_local

    # Crank-slider system Joint index
    jnt_names = [
        "Revolute 10",
        "Revolute 12",
        "Revolute 13",
        "Slider 21"
    ]
    dofs_idx = [Crank_slider_system.get_joint(name).dof_idx_local for name in jnt_names] 

    tablet_link_name = ['tablet']
    tablet = scene.add_entity(
        gs.morphs.MJCF(
            file = "My_asset/Tablet_posmod/Tablet_posmod.xml",
            euler = (90,0,0),
            # Wall : postion : -60, 300, 50
            # motor shaft 최소 좌표: [-120.  340.   10.]
            # motor shaft 최대 좌표: [  0. 400.  90.]
            pos = (-0.5, 3.4, 10.0),
            scale = 10.0,
        )
    )

    print("tablet_link_name : ", tablet_link_name)
    # Tablet link
    tablet_links = [tablet.get_link('tablet') for name in tablet_link_name]
    print("tablet_links : ", tablet_links)
    tablet_link_idx = {tablet_link_name[i]: [None, None] for i in range(len(tablet_link_name))}
    print("tablet_link_idx before : ", tablet_link_idx)
    for i, name in enumerate(tablet_link_name):
        tablet_link_idx[name][0] = tablet_links[i].idx
        tablet_link_idx[name][1] = tablet_links[i].idx_local
    print(tablet_link_idx)

    # add sensors to the scene
    for link_name in tablet_link_name:
        if args.force:
            sensor_options = gs.sensors.ContactForce(
                entity_idx=tablet.idx,
                link_idx_local=tablet.get_link(link_name).idx_local,
                draw_debug=True,
            )
            plot_kwargs = dict(
                title=f"{link_name} Force Sensor Data",
                labels=["force_x", "force_y", "force_z"],
            )
        else:
            sensor_options = gs.sensors.Contact(
                entity_idx=tablet.idx,
                link_idx_local=tablet.get_link(link_name).idx_local,
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

    ## scene build
    scene.build()
    print("------------------- Scene Built ------------------")
    print("Scene Enttities : ", scene.entities)    

    # # Equality constraint
    # link1 = tablet.get_link(tablet_link_name[0])
    # link2 = tablet.get_link(tablet_link_name[1])
    # link1_idx_arr = np.array(link1.idx, dtype=gs.np_int)
    # link2_idx_arr = np.array(link2.idx, dtype=gs.np_int)
    # solver.add_weld_constraint(link1_idx_arr, link2_idx_arr)

    tablet_pos = Crank_slider_system.get_links_pos(link_idx["Wall_1"][0])
    tablet_pos = tablet_pos.tolist()
    tablet_pos[0][2] += 0.05  # Wall 의 두께 고려
    print(tablet_pos)
    # tablet.set_pos(pos = tablet_pos[0])

    # 특정 link 의 좌표를 가져올 수 있는 게 아닌, 전체 Entity 의 좌표를 가져오는 것임.
    print("Wall_position : ", Crank_slider_system.get_links_pos())
    print("Tablet_position : ", tablet.get_links_pos(), tablet.get_pos())
    cam.start_recording()

    ############################### hard reset ##########################
    ######################## control dofs ########################
    Crank_slider_system.set_dofs_kp(
    kp = np.array([1,1,1,1]),
    dofs_idx_local = dofs_idx,
    )
    Crank_slider_system.set_dofs_kv(
        kv = np.array([1,1,1,1]),
        dofs_idx_local = dofs_idx,
    )

    # set_dof_position 
    desired_position = m.pi / 2 
    desired_position_list = [desired_position if i == 0 else 0.0 for i in range(len(dofs_idx))]

    # Crank_slider initial position 설정
    flag = True
    # Crank_slider_system.set_dofs_position(desired_position_list, dofs_idx)
    for i in range(200):
        if flag:
            print("Crank-slider Initial Pos : ", Crank_slider_system.get_dofs_position(dofs_idx))
            flag = False
        # Crank_slider_system.set_dofs_position([desired_position], [0])
        cam.render()
        scene.step()

    # tablet initial position 설정
    tablet_initial_pos = tablet.get_pos().tolist()
    tablet_initial_pos[-1] -= 9.5
    tablet.set_pos(pos = tablet_initial_pos)
    flag = True
    for i in range(200):
        if flag:
            print("Tablet Initial Pos : ", tablet.get_pos())
            flag = False
        cam.render()
        scene.step()

    try:
        # second: 2.0, timestep = 0.01
        steps = int(args.seconds / args.timestep) if "PYTEST_VERSION" not in os.environ else 10
        print("steps : ", steps)
        # cam.set_pose(pos = (5, 3.5, 2.5), lookat = (0, 3.5, 0))

        for _ in range(steps):
            desired_velocity_list = [100.0, 0, 0, 0]                
            Crank_slider_system.set_dofs_velocity(desired_velocity_list, dofs_idx)
            # print(sensor.read())
            cam.render()
            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
        cam.stop_recording(save_to_filename ="video/SystemIntegration_20251111(1).mp4")
        scene.stop_recording()

if __name__ == "__main__":
    main()


# Wall 의 시뮬레이션 상 좌표 : 0.353 0.01 -0.22 .. ?
# Wall_position :  tensor
#       ([[ 0.0000,  0.0000,  0.0000],
#         [-0.1630,  0.1100,  0.0500],
#         [-0.1485,  0.2555,  0.0500],
#         [ 0.0000,  0.0000,  0.0000],
#         [-0.1435,  0.1300,  0.0500],
#         [-0.1435,  0.2240,  0.0500]], device='cuda:0')
# Tablet_position :  tensor
#        ([[0., 0., 0.],
#         [0., 0., 0.]], device='cuda:0')
#
#  [[-0.14350000023841858, 0.12999999523162842, 0.05000000074505806]]


# 2025.10.14 수정 사항
# 크기가 너무 커지거나 작아지면, 시각화 실패하는 줄 알았는데, 아님. 
# 충돌이 발생하면, 시각화 화면이 black out 되는 듯 함.
# fusion 360 기준으로 포지션 지정하는 게 좋음. 무슨 말이냐면, fusion 360의 오리진을 무조건 따라감.
#   - 예를 들어서 (0, 0, 0) 을 기준으로 만들지 않으면 pos 지정이 애매해짐.


# 2025.10.15 해야할것. 
# pyLife 알아보기 -> SN 선도 근사할 아이디어 생각해보기
# tensile strength 식을 convex surface 의 알약에서 근사할 수 있을 듯 함. 
# 이를 통해 구현할 수 있는 부분은 결과적으로 원하는 값은 S-N curve. S-N curve 를 근사할 수 있는 방법 중 tablet braking force 와 tablet tensile strength 를 이용할 수 있음.  
# 위치는 대충 조정 된듯함. motor_shaft_1 의 각도 조정 dof_idx_position ,..? 이런 함수. 

# 2025.10.15 수정사항
# weld 가 풀려버리는 현상이 계속 발생, 공차와 anchor 수정을 통해 해결함.
# anchor = "0 0 0" solref = "0.001 1" solimp = "0.99 0.999 0.001"

# 2025.10.16 수정사항
# weld 불안정성 높음, 따라서 dofs position constrol 로 어느정도 해결해야할듯
# Cranks-slider mechanism passive dofs position 계산 -> CrankSliderMechanism class 생성  