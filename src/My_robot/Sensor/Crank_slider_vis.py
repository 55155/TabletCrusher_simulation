import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import platform

# 한글 폰트 설정 (경고 메시지 제거)
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 크랭크-슬라이더 메커니즘 파라미터
crank_length = 2.0      # 크랭크 길이 (r)
rod_length = 6.0        # 커넥팅 로드 길이 (l)
omega = 2.0             # 각속도 (rad/s)

def calculate_positions(theta):
    """크랭크 각도 theta에서 각 부품의 위치 계산"""
    x_crank = crank_length * np.cos(theta)
    y_crank = crank_length * np.sin(theta)
    x_slider = x_crank + np.sqrt(rod_length**2 - y_crank**2)
    y_slider = 0
    return (0, 0), (x_crank, y_crank), (x_slider, y_slider)

def calculate_kinematics(theta, omega):
    """슬라이더의 속도와 가속도 계산"""
    x_crank = crank_length * np.cos(theta)
    y_crank = crank_length * np.sin(theta)
    x_slider = x_crank + np.sqrt(rod_length**2 - y_crank**2)
    
    v_slider = -crank_length * omega * np.sin(theta) * (1 + crank_length * np.cos(theta) / np.sqrt(rod_length**2 - (crank_length * np.sin(theta))**2))
    
    term1 = -crank_length * omega**2 * np.cos(theta)
    term2 = crank_length * omega**2 * np.sin(theta)**2 / np.sqrt(rod_length**2 - (crank_length * np.sin(theta))**2)
    term3 = crank_length**2 * omega**2 * np.sin(theta)**2 * np.cos(theta) / (rod_length**2 - (crank_length * np.sin(theta))**2)**(3/2)
    a_slider = term1 + term2 + term3
    
    return x_slider, v_slider, a_slider

# 애니메이션 설정
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('크랭크-슬라이더 메커니즘 시뮬레이션', fontsize=16, fontweight='bold')

# 상단: 메커니즘 동작
ax1.set_xlim(-1, crank_length + rod_length + 1)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X 위치 (m)', fontsize=12)
ax1.set_ylabel('Y 위치 (m)', fontsize=12)
ax1.set_title('메커니즘 동작', fontsize=14)

# 메커니즘 요소
crank_line, = ax1.plot([], [], 'b-', linewidth=4, label='크랭크', zorder=5)
rod_line, = ax1.plot([], [], 'r-', linewidth=4, label='커넥팅 로드', zorder=5)
crank_joint = Circle((0, 0), 0.15, color='black', zorder=10)
crank_pin = Circle((0, 0), 0.12, color='blue', zorder=10)
slider_pin = Circle((0, 0), 0.12, color='red', zorder=10)
slider_rect = Rectangle((0, -0.3), 0.8, 0.6, color='green', alpha=0.7, zorder=5)

ax1.add_patch(crank_joint)
ax1.add_patch(crank_pin)
ax1.add_patch(slider_pin)
ax1.add_patch(slider_rect)
ax1.plot([-1, crank_length + rod_length + 1], [0, 0], 'k--', linewidth=1, alpha=0.5)
ax1.legend(loc='upper right', fontsize=10)

# 하단: 운동학 그래프 (동적 업데이트)
ax2.set_xlim(0, 10)
ax2.set_ylim(-20, 20)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('시간 (s)', fontsize=12)
ax2.set_ylabel('값', fontsize=12)
ax2.set_title('슬라이더 운동학 (실시간)', fontsize=14)

pos_line, = ax2.plot([], [], 'b-', linewidth=2.5, label='위치 (m)', alpha=0.8)
vel_line, = ax2.plot([], [], 'g-', linewidth=2.5, label='속도 (m/s)', alpha=0.8)
acc_line, = ax2.plot([], [], 'r-', linewidth=2.5, label='가속도 (m/s²)', alpha=0.8)
ax2.legend(loc='upper right', fontsize=10)

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=11,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 데이터 저장
time_data = []
position_data = []
velocity_data = []
acceleration_data = []

def update(frame):
    """애니메이션 업데이트 함수 - 매 프레임마다 호출"""
    t = frame * 0.05
    theta = omega * t
    
    # 위치 계산 및 업데이트
    origin, crank_end, slider_pos = calculate_positions(theta)
    crank_line.set_data([origin[0], crank_end[0]], [origin[1], crank_end[1]])
    rod_line.set_data([crank_end[0], slider_pos[0]], [crank_end[1], slider_pos[1]])
    crank_pin.center = crank_end
    slider_pin.center = slider_pos
    slider_rect.set_x(slider_pos[0] - 0.4)
    
    # 운동학 계산
    x_slider, v_slider, a_slider = calculate_kinematics(theta, omega)
    
    # 데이터 누적
    time_data.append(t)
    position_data.append(x_slider)
    velocity_data.append(v_slider)
    acceleration_data.append(a_slider)
    
    # 최근 200개 데이터만 표시
    window = 200
    display_time = time_data[-window:]
    display_pos = position_data[-window:]
    display_vel = velocity_data[-window:]
    display_acc = acceleration_data[-window:]
    
    # 그래프 데이터 업데이트 (중요!)
    pos_line.set_data(display_time, display_pos)
    vel_line.set_data(display_time, display_vel)
    acc_line.set_data(display_time, display_acc)
    
    # x축 동적 조정
    if t > 10:
        ax2.set_xlim(t - 10, t + 0.5)
    
    # y축 동적 조정
    if len(display_pos) > 10:
        all_vals = display_pos + display_vel + display_acc
        margin = 2
        ax2.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
    
    # 시간 표시
    time_text.set_text(f'시간: {t:.2f} s\n각도: {np.degrees(theta) % 360:.1f}°\n위치: {x_slider:.2f} m')
    
    return [crank_line, rod_line, crank_pin, slider_pin, slider_rect, 
            pos_line, vel_line, acc_line, time_text]

# 애니메이션 생성 (blit=False로 안정적 업데이트)
anim = animation.FuncAnimation(fig, update, frames=600, interval=50, 
                               blit=False, repeat=True)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("크랭크-슬라이더 메커니즘 시뮬레이션 실행 중...")
print("="*60)
print(f"크랭크 길이: {crank_length} m")
print(f"커넥팅 로드 길이: {rod_length} m")
print(f"각속도: {omega} rad/s ({np.degrees(omega):.1f} deg/s)")
print(f"스트로크: {2*crank_length} m")
print(f"주기: {2*np.pi/omega:.2f} s")
print("="*60)
