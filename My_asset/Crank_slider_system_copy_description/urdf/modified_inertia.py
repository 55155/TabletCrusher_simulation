import xml.etree.ElementTree as ET

# 사용자 URDF 파일 경로 지정
input_urdf_path = './Crank_slider_system_copy.urdf'  # 여기에 본인 URDF 파일 경로 적기
output_urdf_path = './Crank_slider_system_copy_modified.urdf'  # 수정 후 저장할 파일 경로

MIN_INERTIA_VALUE = 1e-6  # 최소 관성값 (0보다 작은 값 대체용)

def fix_inertia_values(urdf_path, output_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 모든 <inertial> 태그 안의 <inertia> 검사
    for inertial in root.findall('.//inertial'):
        inertia = inertial.find('inertia')
        if inertia is not None:
            ixx = float(inertia.get('ixx', '0'))
            iyy = float(inertia.get('iyy', '0'))
            izz = float(inertia.get('izz', '0'))

            # 0 이하 값은 MIN_INERTIA_VALUE로 변경
            if ixx <= 0:
                inertia.set('ixx', str(MIN_INERTIA_VALUE))
            if iyy <= 0:
                inertia.set('iyy', str(MIN_INERTIA_VALUE))
            if izz <= 0:
                inertia.set('izz', str(MIN_INERTIA_VALUE))

    # 수정된 URDF 저장
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

# 함수 실행 예시
fix_inertia_values(input_urdf_path, output_urdf_path)
print(f"Modified URDF saved to {output_urdf_path}")
