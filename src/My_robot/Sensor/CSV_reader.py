import csv
from pathlib import Path
import numpy as np

# CSV 파일 경로
ROBOT_FORCE_PATH = Path("robot_actuation_forces.csv")
SENSOR_FORCE_PATH = Path("sensor_contact_forces.csv")

def init_robot_csv(path=ROBOT_FORCE_PATH):
    """로봇 4개 joint actuation force용 CSV 초기화"""
    is_new = not path.exists()
    f = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["time", "joint0_force", "joint1_force", "joint2_force", "joint3_force"])
    return f, writer

def init_sensor_csv(path=SENSOR_FORCE_PATH):
    """충돌체 sensor force용 CSV 초기화 (3D vector)"""
    is_new = not path.exists()
    f = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["time", "sensor_fx", "sensor_fy", "sensor_fz", "sensor_force_magnitude"])
    return f, writer

def log_robot_forces(writer, t, forces):
    """로봇 4개 joint force 로깅"""
    writer.writerow([t] + forces.tolist())

def log_sensor_force(writer, t, sensor_force):
    """sensor 3D force 로깅 (magnitude도 추가)"""
    fx, fy, fz = sensor_force  # [x, y, z]
    magnitude = np.linalg.norm(sensor_force)
    writer.writerow([t, fx, fy, fz, magnitude])

# 메인 시뮬레이션 루프에 통합
try:
    # CSV 파일들 초기화
    robot_f, robot_writer = init_robot_csv()
    sensor_f, sensor_writer = init_sensor_csv()
    
    steps = int(args.seconds / args.timestep) if "PYTEST_VERSION" not in os.environ else 10
    print(f"Total steps: {steps}")
    
    box.set_pos(pos=box_initial_pos)
    t = 0.0
    
    for step in range(steps):
        # 1. 제어 및 시뮬레이션 스텝
        Crank_slider_system.control_dofs_force(desired_force_list, [0])
        
        # 2. 현재 상태 로깅 (CSV에 저장)
        current_time = t
        
        # 로봇 4개 joint actuation force 저장
        robot_forces = Crank_slider_system.get_dofs_force()  # shape: (4,)
        log_robot_forces(robot_writer, current_time, robot_forces)
        
        # sensor force 저장 (충돌체 힘)
        sensor_force = sensor.read()  # shape: (3,) 예상
        log_sensor_force(sensor_writer, current_time, sensor_force)
        
        # 3. 디버깅 출력 (선택적)
        print(f"Step {step}: Robot forces={robot_forces.tolist()}, "
              f"Sensor force={sensor_force.tolist()}, magnitude={np.linalg.norm(sensor_force):.2f}")
        
        # 4. 렌더링
        cam.set_pose(lookat=(-0.5, 3.4, .5), pos=(1, 3.4, .5))
        cam.render()
        scene.step()
        
        t += args.timestep

except KeyboardInterrupt:
    gs.logger.info("Simulation interrupted, exiting.")
finally:
    # CSV 파일 안전하게 닫기
    robot_f.close()
    sensor_f.close()
    gs.logger.info("Simulation finished. CSV files saved:")
    gs.logger.info(f"  - Robot forces: {ROBOT_FORCE_PATH}")
    gs.logger.info(f"  - Sensor forces: {SENSOR_FORCE_PATH}")
    cam.stop_recording(save_to_filename="video/[20260127]SystemIntegration (2).mp4")
    scene.stop_recording()
