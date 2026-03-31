# import os

# # OBJ 파일이 있는 폴더 경로 (실제 경로로 수정)
# folder_path = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_V3_Pjoint_description/urdf/"

# # 폴더 내 모든 .obj 파일 리스트
# obj_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.obj')]

# # scale 파라미터 (필요시 수정)
# scale_str = '0.001 0.001 0.001'

# # 출력할 문자열 리스트
# output_lines = []
# for obj_file in sorted(obj_files):
#     abs_path = os.path.join(folder_path, obj_file)
#     Temp = os.path.splitext(obj_file)[0]
#     output_lines.append(f'<mesh name="{Temp}" file="{abs_path}" scale="{scale_str}"/>')

# for obj_file in sorted(obj_files):
#     abs_path = os.path.join(folder_path, obj_file)
#     Temp = os.path.splitext(obj_file)[0]
#     # <geom rgba="0.7 0.7 0.7 1" type="mesh" mesh="Slider1_1_collision_(182)"/>
    
#     # 색상 바꾸는 것 고려
#     output_lines.append(f'<geom rgba="0.7 0.7 0.7 1" type="mesh" mesh="{Temp}"/>')

# # txt 파일로 저장
# output_path = os.path.join(folder_path, 'mesh_list.txt')
# with open(output_path, 'w') as f:
#     f.write('\n'.join(output_lines))


import os
import colorsys

# Tablet만 다양한 색상으로 적용
folder_path = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet_posmod/"

obj_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.obj')]
scale_str = '0.001 0.001 0.001'

output_lines = []

# mesh 태그 생성
for obj_file in sorted(obj_files):
    abs_path = os.path.join(folder_path, obj_file)
    Temp = os.path.splitext(obj_file)[0]
    output_lines.append(f'<mesh name="{Temp}" file="{abs_path}" scale="{scale_str}"/>')

# 15개의 색상 생성 (H: 색조, S: 채도, L: 명도)
# 색조는 0~1을 균일하게 분할, 채도는 0.3~1.0 범위에서 점진적으로 다르게 설정함, 명도는 중간값 0.5 고정
num_colors = 15
hue_vals = [i / num_colors for i in range(num_colors)]
sat_vals = [0.3 + 0.7 * (i / (num_colors - 1)) for i in range(num_colors)]
light_val = 0.5

# obj 파일 개수 만큼 반복, 색상 인덱스는 모듈로 처리
for i, obj_file in enumerate(sorted(obj_files)):
    abs_path = os.path.join(folder_path, obj_file)
    Temp = os.path.splitext(obj_file)[0]

    # HSL -> RGB 변환
    h = hue_vals[i % num_colors]
    s = sat_vals[i % num_colors]
    l = light_val
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    # rgba 형식 (맨 끝에 알파 1)
    rgba_str = f"{r:.3f} {g:.3f} {b:.3f} 1"

    output_lines.append(f'<geom rgba="{rgba_str}" type="mesh" mesh="{Temp}"/>')

output_path = os.path.join(folder_path, 'mesh_list.txt')
with open(output_path, 'w') as f:
    f.write('\n'.join(output_lines))
