import argparse
import os
import numpy as np  
from tqdm import tqdm

# import roma
import torch
import math as m

# # 오일러 각을 회전 행렬로 변환
# euler_angles = [90, 0, 90]  # degrees
# R = roma.euler_to_rotmat('XYZ', euler_angles, degrees=True)

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE
from pathlib import Path

# CSV 파일 경로
ROBOT_FORCE_PATH = Path("robot_actuation_forces.csv")
ROBOT_VELOCITY_PATH = Path("robot_joint_velocities.csv")
SENSOR_FORCE_PATH = Path("sensor_contact_forces.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=0.001, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization GUI")
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=5.0, help="Number of seconds to simulate")
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
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            # constraint_timeconst -> weld 판단 
            # constraint_timeconst=max(0.01, 2 * args.timestep),
            use_gjk_collision=True,
            # enable_collision=False,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=20,
            res=(960, 1080),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )
    cam = scene.add_camera(
        res=(1280, 960),
        # scale=10.0
        # pos = (10,2,5),

        # scale=5.0
        # pos = (5,1,2.5),
        # lookat=(0, 1, 0.0),
        # fov=30,

        # scale=1.0
        pos = (1, 0.5, 1.0),
        lookat=(0, 0.2, 0.0),
        
        GUI=True,
    )

    # rigid solver : for add_constaraints
    solver = scene.sim.rigid_solver
    # Crank-slider system
    Crank_slider_system = scene.add_entity(
        gs.morphs.MJCF(
            file = "./My_asset/Crusher_description2/urdf/" \
            "Crusher.xml",
            pos = (0, 0.0, 0),
            scale = 1.0,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
        visualize_contact=True,
    )

    link_name = [
    "motor_shaft_1",
    "Link2_1",
    "Link3_1",
    "shaft_1",
    # "Wall_1"
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
        # "Revolute 12",
        # "Revolute 13",
        # "Slider 21"
    ]
    dofs_idx = [Crank_slider_system.get_joint(name).dof_idx_local for name in jnt_names] 

    tablet_link_name = ['tablet']
    tablet = scene.add_entity(
        gs.morphs.MJCF(
            file = "My_asset/Tablet_posmod/Tablet_posmod.xml",
            euler = (90,0,0),
            # scale = 10.0일 때의 기준
            # Wall : postion : -60, 300, 50
            # motor shaft 최소 좌표: [-120.  340.   10.]
            # motor shaft 최대 좌표: [  0. 400.  90.]
            # pos = (-0.5, 3.4, 10.0),
            
            # scale = 5.0일 때의 기준
            # pos = (-.15, 1.65, -10.0),           
            # scale = 5.0,

            # scale = 1.0일 때의 기준
            pos = (-0.03, 0.33, -10.0),
            scale = 1.0,
        )
    )
    tablet_freejoint = scene.add_entity(
        gs.morphs.MJCF(
            file = "My_asset/Tablet_posmod/Tablet_posmod_freejoint.xml",
            euler = (90,0,0),
            pos = (0, -1, 0),
            scale = 5.0,
        )
    )

    ## Test 용 box ( Talbet 대체 )
    box = scene.add_entity(
        gs.morphs.Box(
            pos = (-0.03, 0.33, -10.0),
            # pos = (-.15, 1.65, 10.0),
            # pos = (-0.5, 3.2, 10.0),
            # scale = 5.0,
            # size = (0.05, 0.02, 0.05),
            # scale= 1.0,
            size = (0.01, 0.01, 0.01),
            fixed = True,
        )
    )
    plane = scene.add_entity(
        gs.morphs.Plane(
            pos = (0.0, 0.0, 0.0),
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

    # # add sensors to the scene
    # for link_name in tablet_link_name:
    #     if args.force:
    #         sensor_options = gs.sensors.ContactForce(
    #             entity_idx=tablet.idx,
    #             link_idx_local=tablet.get_link(link_name).idx_local,
    #             draw_debug=True,
    #         )
    #         plot_kwargs = dict(
    #             title=f"{link_name} Force Sensor Data",
    #             labels=["force_x", "force_y", "force_z"],
    #         )
    #     else:
    #         sensor_options = gs.sensors.Contact(
    #             entity_idx=tablet.idx,
    #             link_idx_local=tablet.get_link(link_name).idx_local,
    #             draw_debug=True,
    #         )
    #         plot_kwargs = dict(
    #             title=f"{link_name} Contact Sensor Data",
    #             labels=["in_contact"],
    #             window_size=(960, 1080),
    #         )

    # box Entity의 경우에 문제가 생기는지 확인. 
    if args.force:
        sensor_options = gs.sensors.ContactForce(
            entity_idx=box.idx,
            draw_debug=True,
        )
        plot_kwargs = dict(
            title=f"{link_name} Force Sensor Data",
            labels=["force_x", "force_y", "force_z"],
            window_size=(960, 1080),
        )
    else:
        pass

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


    # 특정 link 의 좌표를 가져올 수 있는 게 아닌, 전체 Entity 의 좌표를 가져오는 것임.
    print("Wall_position : ", Crank_slider_system.get_links_pos())
    print("Tablet_position : ", tablet.get_links_pos(), tablet.get_pos())
    cam.start_recording()

    ############################### hard reset ##########################
    ######################## control dofs ########################
    Crank_slider_system.set_dofs_kp(
        kp = np.array([0.1,]),
        dofs_idx_local = [0],
    )
    Crank_slider_system.set_dofs_kv(
        kv = np.array([.1,]),
        dofs_idx_local = [0],
    )
    Crank_slider_system.set_dofs_armature(
        armature = np.array([0.05,]),
        dofs_idx_local = [0],
    )
    # set_dof_position 
    desired_position = -0.5 * m.pi 
    desired_position_list = [desired_position if i == 0 else 0.0 for i in range(len(dofs_idx))]

    # Crank_slider initial position 설정
    flag = True
    Crank_slider_system.set_dofs_position(desired_position_list, dofs_idx)
    for i in range(1000):    
        if flag:
            print("Crank-slider Initial Pos : ", Crank_slider_system.get_dofs_position(dofs_idx))
            flag = False
        Crank_slider_system.control_dofs_position(desired_position_list, [0])
        cam.render()
        scene.step()

    # tablet initial positin 설정
    tablet_initial_pos = tablet.get_pos().tolist()
    tablet_update_pos = tablet_initial_pos.copy()
    tablet_update_pos[-1] += 10.05  # Wall 두께 고려
    box.set_pos(pos = tablet_update_pos)

    # box initial position 설정
    # box_initial_pos = box.get_pos().tolist()
    # box_initial_pos[-1] -= 9.8
    # box.set_pos(pos = box_initial_pos)
    flag = True
    for i in range(200):
        if flag:
            print("Tablet Initial Pos : ", tablet.get_pos())
            print("Box Initial Pos : ", box.get_pos())
            flag = False
        cam.render()
        scene.step()

    import CSV_reader # CSV 로깅 유틸리티
    from collections import deque
    
    ZERO_WINDOW = 50  # 최근 스텝에서 힘이 0인지 판단하는 윈도우 크기
    EPS = 0.5  # 힘이 0으로 간주되는 임계값

    # 최근 100스텝의 "크랭크 힘" 기록 (명령값 기준)
    crank_force_q = deque(maxlen=ZERO_WINDOW)
    crank_velocity_q = deque(maxlen=ZERO_WINDOW)

    direction = 1.0             # +1이면 정방향, -1이면 역방향
    force_amp = 15.0            # 힘 크기 (원하는 대로)
    velocity_amp = 16 * m.pi / 60      # 속도 크기 (원하는 대로)
    desired_force_list = [direction * force_amp]
    desired_velocity_list = [direction * velocity_amp]
    try:
        robot_f, robot_writer = CSV_reader.init_robot_csv()
        robot_vel_f, robot_vel_writer = CSV_reader.init_robot_velocity_csv()
        sensor_f, sensor_writer = CSV_reader.init_sensor_csv()

        # second: 2.0, timestep = 0.001
        steps = int(args.seconds / args.timestep) if "PYTEST_VERSION" not in os.environ else 10
        print("steps : ", steps)
        desired_force_list = [force_amp]

        for _ in range(steps):
            # 1. 제어 및 robot forces 저장
            # 1-1. force control
            Crank_slider_system.control_dofs_force(desired_force_list, [0])
            # 1-2. velocity control
            # Crank_slider_system.control_dofs_velocity(desired_velocity_list, [0])
            # 1-3. robot force 로그
            robot_forces = Crank_slider_system.get_dofs_force().cpu().numpy()
            CSV_reader.log_robot_forces(robot_writer, _, robot_forces)
            # 1-4 velocity 로그
            robot_velocities = Crank_slider_system.get_dofs_velocity().cpu().numpy()
            CSV_reader.log_robot_velocities(robot_vel_writer, _, robot_velocities)

            # 2. Sensor force 저장
            # 2-1. contact sensor
            sensor_force = sensor.read().cpu().numpy()  # shape: (3,) 예상
            CSV_reader.log_sensor_force(sensor_writer, _, sensor_force)
        
            # 3. 디버깅 출력 (선택적)
            print(f"Step {_}: Robot forces={robot_forces.tolist()}, "
                f"Sensor force={sensor_force.tolist()}, magnitude={np.linalg.norm(sensor_force):.2f}")

            # 4-1. 최근 크랭크 힘 기록 업데이트 (힘 제어 시)
            crank_force_q.append(robot_forces[0])  # 크랭크 힘만 기록, queue 에 업데이트
            if len(crank_force_q) == ZERO_WINDOW and all(abs(x) < EPS for x in crank_force_q):
                direction *= -1.0
                desired_force_list = [direction * force_amp]
                crank_force_q.clear() # force queue 도 초기화

            # 4-2 최근 크랭크 속도 기록 업데이트 (속도 제어 시)
            # crank_velocity_q.append(robot_velocities[0])  # 크랭크 속도만 기록, queue 에 업데이트
            # if len(crank_velocity_q) == ZERO_WINDOW and all(abs(x) < EPS for x in crank_velocity_q):
            #     direction *= -1.0
            #     desired_velocity_list = [direction * velocity_amp]
            #     crank_velocity_q.clear() # velocity queue 도 초기화

            # 카메라-타블렛 위치
            # cam.set_pose(
            #     lookat = (-0.25, 1.65, .25),
            #     pos=(1.0, 1.65, .25),
            # )
            cam.set_pose()
            cam.render()
            scene.step()
            # 실제 파손 모델링 적용 constraint weld 해제 
            if _ == steps // 2:
                # tablet.set_pos(pos=tablet_initial_pos)
                # tablet_freejoint.set_pos(pos=tablet_update_pos)
                print(tablet_initial_pos)
                print(tablet_update_pos)

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        robot_f.close()
        robot_vel_f.close()
        sensor_f.close()
        gs.logger.info("Simulation finished.")
        gs.logger.info(f"  - Robot forces: {ROBOT_FORCE_PATH}")
        gs.logger.info(f"  - Sensor forces: {SENSOR_FORCE_PATH}")
        cam.stop_recording(save_to_filename ="video/[20260223]Tablet충돌힘계산_Pjoint알약.mp4")
        scene.stop_recording()

if __name__ == "__main__":
    main()


# Wall 의 시뮬레이션 상 좌표 : 0.353 0.01 -0.22 .. ?
# Wall_position :  tensor
#       ([[ 0.0000,  0.0000,  0.0000]
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


# 2026.01.26 수정사항
# 시스템 통합 2차 시도
# https://github.com/Genesis-Embodied-AI/Genesis/issues/1993
# 