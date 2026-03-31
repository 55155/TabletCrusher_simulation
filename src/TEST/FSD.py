import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue

# 시리얼 데이터를 저장할 큐 생성
data_queue = queue.Queue()

# 상수 정의
Rm = 650  # 출력저항 기준 옴
Vin = 5.0  # 입력 전압 V

def serial_read_thread(port, baudrate, data_queue):
    ser = serial.Serial(port, baudrate, timeout=0.1)
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                try:
                    raw = float(line)
                    if raw == 0:
                        continue
                    # 0~1023 ADC 값을 0~5V 범위로 정규화
                    Vout = (raw / 1023.0) * Vin
                    if Vout > Vin:
                        continue
                    Rout = Rm * (Vin - Vout) / Vout
                    data_queue.put(Rout)
                except ValueError:
                    pass
        except serial.SerialException:
            break


def animate(i, xs, ys, data_queue):
    while not data_queue.empty():
        data = data_queue.get()
        xs.append(i)
        ys.append(data)

    xs = xs[-100:]
    ys = ys[-100:]

    plt.cla()
    plt.plot(xs, ys)
    plt.xlabel('Time')
    plt.ylabel('Output Resistance (Ω)')

    if ys:
        min_val = min(ys)
        plt.title(f'Real-time Output Resistance Plot from Arduino\nMin: {min_val:.2f} Ω')
    else:
        plt.title('Real-time Output Resistance Plot from Arduino')

    plt.grid(True)
    plt.tight_layout()

    return xs, ys


if __name__ == '__main__':
    port = '/dev/cu.usbmodem1401'  # Mac 환경에 맞게 수정
    baudrate = 9600

    xs, ys = [], []

    # 시리얼 읽기용 백그라운드 스레드 시작
    thread = threading.Thread(target=serial_read_thread, args=(port, baudrate, data_queue))
    thread.daemon = True
    thread.start()

    # matplotlib 애니메이션 시작
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys, data_queue), interval=100, save_count=100)

    # ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys, data_queue), interval=100)

    plt.show()
