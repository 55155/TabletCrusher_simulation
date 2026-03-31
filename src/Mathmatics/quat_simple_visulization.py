import numpy as np
import matplotlib.pyplot as plt
from quat import *

if __name__ == "__main__":
    v = np.array([0,0,1])          # 처음 벡터 (예시)
    rotation = np.array([-1, -3, 2]) # y축
    rotation_axis = normalize_quaternion(rotation)
    rotation_deg = 90
    rotation_rad = degree_to_radian(rotation_deg)
    rotated_v = rotate_vector(v, rotation_axis, rotation_rad)
    
    print(rotated_v)
    # 시각화 (3D 화살표)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 화살표 (원점에서 회전 후 벡터)
    ax.quiver(0, 0, 0, rotated_v[0], rotated_v[1], rotated_v[2], 
              color='b', linewidth=3, arrow_length_ratio=0.18)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Quaternion Rotation Visualization")
    plt.tight_layout()
    plt.show()