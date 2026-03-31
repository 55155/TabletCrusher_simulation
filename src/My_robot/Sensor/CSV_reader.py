import csv
from pathlib import Path
import numpy as np

# CSV 파일 경로
ROBOT_FORCE_PATH = Path("robot_actuation_forces.csv")
SENSOR_FORCE_PATH = Path("sensor_contact_forces.csv")
ROBOT_VELOCITY_PATH = Path("robot_joint_velocities.csv")

def init_robot_csv(path=ROBOT_FORCE_PATH):
    """로봇 4개 joint actuation force용 CSV 초기화"""
    is_new = not path.exists()
    f = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["time", "joint0_force", "joint1_force", "joint2_force", "joint3_force"])
    return f, writer

def init_robot_velocity_csv(path=ROBOT_VELOCITY_PATH):
    """로봇 4개 joint actuation velocity용 CSV 초기화"""
    is_new = not path.exists()
    f = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["time", "joint0_velocity", "joint1_velocity", "joint2_velocity", "joint3_velocity"])
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

def log_robot_velocities(writer, t, velocities):
    """로봇 4개 joint velocity 로깅"""
    writer.writerow([t] + velocities.tolist())

def log_sensor_force(writer, t, sensor_force):
    """sensor 3D force 로깅 (magnitude도 추가)"""
    fx, fy, fz = sensor_force  # [x, y, z]
    magnitude = np.linalg.norm(sensor_force)
    writer.writerow([t, fx, fy, fz, magnitude])
    
if __name__ == "__main__":
    import sys
    import pandas as pd
    import numpy as np
    from PyQt6 import QtWidgets, QtCore
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore as pg_QtCore  # pyqtgraph 내부 호환

    class ForcePlotter(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Robot & Sensor Force Plot - Genesis Data (PyQt6)')
            self.setGeometry(100, 100, 1400, 900)
            # 파일별 다른 컬럼 설정
            self.robot_force_cols = ['joint0_force', 'joint1_force', 'joint2_force', 'joint3_force']
            self.sensor_force_cols = ['sensor_fx', 'sensor_fy', 'sensor_fz', 'sensor_force_magnitude']
            self.step_col = 'time'
            # Central widget
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            
            # Main layout
            layout = QtWidgets.QVBoxLayout(central)
            
            # Plot widget (고성능 PyQt6)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('#f0f0f0')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setMinimumHeight(600)
            layout.addWidget(self.plot_widget)
            
            # Control panel
            ctrl_layout = QtWidgets.QHBoxLayout()
            
            self.load_robot_btn = QtWidgets.QPushButton('📁 Load robot_actuation_forces.csv')
            self.load_sensor_btn = QtWidgets.QPushButton('📁 Load sensor_contact_forces.csv')
            self.clear_btn = QtWidgets.QPushButton('🗑️ Clear All')
            
            self.load_robot_btn.clicked.connect(self.load_robot)
            self.load_sensor_btn.clicked.connect(self.load_sensor)
            self.clear_btn.clicked.connect(self.clear_plots)
            
            ctrl_layout.addWidget(self.load_robot_btn)
            ctrl_layout.addWidget(self.load_sensor_btn)
            ctrl_layout.addStretch()
            ctrl_layout.addWidget(self.clear_btn)
            
            layout.addLayout(ctrl_layout)
            
            # Status bar
            self.statusBar().showMessage('Ready - Load CSV files to plot')
            
            # Legend & 데이터
            self.legend = self.plot_widget.addLegend(size=(150, 80), offset=(-200, 20))
            self.robot_data = None
            self.sensor_data = None
            self.curves = []  # 플롯 관리
            
            # 컬럼 설정 (CSV에 맞게 수정)
            self.step_col = 'time'  # 또는 'time', 't'
            self.force_cols = ['force_x', 'force_y', 'force_z']  # 실제 컬럼명
            
        def load_robot(self):
            file, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open robot_force.csv', '', 'CSV (*.csv)')
            if file:
                try:
                    self.robot_data = pd.read_csv(file, header=0)  # 헤더 있음!
                    print("Robot CSV columns:", self.robot_data.columns.tolist())  # 확인용
                    self.statusBar().showMessage(f'Loaded robot_force: {len(self.robot_data)} steps')
                    self.plot_data()
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, 'Error', f'Failed to load: {e}')

        def load_sensor(self):
            file, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open sensor_force.csv', '', 'CSV (*.csv)')
            if file:
                try:
                    self.sensor_data = pd.read_csv(file, header=0)  # 헤더 있음!
                    print("Sensor CSV columns:", self.sensor_data.columns.tolist())
                    self.statusBar().showMessage(f'Loaded sensor_force: {len(self.sensor_data)} steps')
                    self.plot_data()
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, 'Error', f'Failed to load: {e}')

        
        def clear_plots(self):
            for curve in self.curves:
                self.plot_widget.removeItem(curve)
            self.curves.clear()
            self.statusBar().showMessage('Plots cleared')
        
        def plot_data(self):
            self.plot_widget.clear()
            self.curves.clear()
            
            # Robot 데이터 (joint forces)
            if self.robot_data is not None:
                steps = self.robot_data[self.step_col].values
                colors = ['r', 'g', 'orange', 'm']
                for i, col in enumerate(self.robot_force_cols):
                    if col in self.robot_data.columns:
                        y = self.robot_data[col].values
                        curve = self.plot_widget.plot(steps, y, 
                            pen=pg.mkPen(color=colors[i], width=2), 
                            name=f'Robot Joint{i}', connect='finite')
                        self.curves.append(curve)
            
            # Sensor 데이터 (sensor forces)
            if self.sensor_data is not None:
                steps = self.sensor_data[self.step_col].values
                colors = ['b', 'c', 'y', 'k']
                for i, col in enumerate(self.sensor_force_cols):
                    if col in self.sensor_data.columns:
                        y = self.sensor_data[col].values
                        curve = self.plot_widget.plot(steps, y, 
                            pen=pg.mkPen(color=colors[i], width=2), 
                            name=f'Sensor {col}', connect='finite')
                        self.curves.append(curve)
            
            self.plot_widget.setLabel('left', 'Force (N)')
            self.plot_widget.setLabel('bottom', 'Time Step')
            self.plot_widget.autoRange()


    app = QtWidgets.QApplication(sys.argv)
    
    # PyQtGraph 설정 (PyQt6 최적화)
    pg.setConfigOptions(antialias=True, foreground='k', background='#f0f0f0')
    
    window = ForcePlotter()
    window.show()
    sys.exit(app.exec())


