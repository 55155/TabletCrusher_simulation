import numpy as np

def quaternion_multiply(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero quaternion cannot be normalized")
    return q / norm

def degree_to_radian(degree: float) -> float:
    return degree * (np.pi / 180.0)

def rotate_vector(v, axis, angle_rad):

    w = np.cos(angle_rad / 2)
    x, y, z = axis * np.sin(angle_rad / 2)

    q = np.array([w, x, y, z])
    q_conj = np.array([w, -x, -y, -z])
    v_q = np.array([0] + list(v))

    rotated_q = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
    return rotated_q[1:]  # 벡터 부분만 반환

if __name__ == "__main__":
    # 테스트 코드
    # 예시: z축 기준 90도 회전
    v = np.array([0,0,1])
    rotation = np.array([0, 1, 0])
    rotation_axis = normalize_quaternion(rotation)
    rotation_deg = 180
    rotation_rad = degree_to_radian(rotation_deg)

    rotated_v = rotate_vector(v, rotation_axis, rotation_rad)
    print(rotated_v)  # 결과: [0. 1. 0.]
