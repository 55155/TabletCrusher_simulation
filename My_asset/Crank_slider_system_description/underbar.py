import os

root = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Crank_slider_system_description/"
directorys = [root + '/vis_mesh', root + '/col_meshes']

for directory in directorys:
    for filename in os.listdir(directory):
        if filename == "base_link":
            continue

        # 확장자 분리
        name, ext = os.path.splitext(filename)

        # '_1'만 제거 (언더바 포함하여)
        new_name_part = name.replace('_1', '')

        # 새 파일명 생성
        new_name = new_name_part + ext

        if new_name != filename:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
