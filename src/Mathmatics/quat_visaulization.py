import numpy as np
import matplotlib.pyplot as plt
from quat import *

# --- 원기둥 생성 및 회전 시각화 (Z축 수직) ---
def plot_rotated_cylinder(axis, angle_deg):
    n = 80     # 점 개수
    h = np.linspace(-1, 1, n)
    theta = np.linspace(0, 2 * np.pi, n)
    H, Theta = np.meshgrid(h, theta)
    X = np.cos(Theta)
    Y = np.sin(Theta)
    Z = H  # Z축 수직

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    axis_norm = axis / np.linalg.norm(axis)
    angle_rad = angle_deg * (np.pi / 180)

    rotated_points = np.array([
        rotate_vector(p, axis_norm, angle_rad) for p in points
    ])
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, color='blue', alpha=0.4, s=8, label='원기둥 (회전 전)')
    ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
               color='red', alpha=0.5, s=10, label='원기둥 (회전 후)')
    axis_len = 1.5
    ax.quiver(0, 0, 0, axis_norm[0]*axis_len, axis_norm[1]*axis_len, axis_norm[2]*axis_len,
              color='green', linewidth=3, label='회전축')
    ax.legend()
    ax.set_title('쿼터니언 회전에 의한 Z축 수직 원기둥 3D 시각화')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

# 예시 실행: axis = (0,1,1), angle_deg=180
plot_rotated_cylinder(np.array([1, 0, 0]), 90)
