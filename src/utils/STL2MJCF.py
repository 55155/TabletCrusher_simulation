import coacd
import trimesh
import yaml
import os

def get_all_stl_files(directory):
    stl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.stl'):
                stl_files.append(os.path.join(root, file))
    return stl_files

with open("/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/src/config.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.Loader)

asset_path = config['file_path']['asset']

def stl_to_obj(stl_path, obj_path):
    mesh = trimesh.load_mesh(stl_path)
    mesh.export(obj_path, file_type='obj')
    print(f"Converted '{stl_path}' to '{obj_path}'")

stl_path = asset_path + '/Tablet_posmod'
obj_path = asset_path + '/Tablet_posmod'

stl_files = get_all_stl_files(stl_path)
for stl_file in stl_files:
    obj_file = stl_file.split('.')[0]
    stl_to_obj(stl_file, obj_file + ".obj")

    input_file = obj_file + ".obj"
    print(input_file)

    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh)  # a list of convex hulls.

    print(type(parts), type(parts[0]), parts[0] )

    # 각 분할된 볼록 메쉬 저장 추가
    for i, part in enumerate(parts):
        vertices, faces = part  # 리스트 구조 해제
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        output_obj_path = f"{obj_file}_collision_({i+1}).obj"
        part_mesh.export(output_obj_path)
        print(f"Saved convex hull part {i+1} to {output_obj_path}")