import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 안정적인 matplotlib backend 설정 (Mac/Linux 호환)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import pi, sin, cos, sqrt, arctan2, arctan
from typing import Tuple, Optional, Literal, Dict


class CrankSlider:
    """
    Crank-Slider Mechanism Class
    
    Parameters
    ----------
    Crank_length : float
        Crank length
    rod_length : float
        Connecting rod (coupler) length
    offset : float, optional
        Slider offset from crank axis (0 for in-line, non-zero for offset)
    crank_position : tuple, optional
        Crank pivot position (x, y)
    slider_angle : float, optional
        Slider axis angle in radians (0 for horizontal)
    """
    
    def __init__(
        self,
        Crank_length: float,
        rod_length: float,
        offset: float = 0.0,
        crank_position: Tuple[float, float] = (0, 0),
        slider_angle: float = 0.0
    ):
        self.Crank_length = Crank_length
        self.rod_length = rod_length
        self.offset = offset
        self.x_crank, self.y_crank = crank_position
        self.slider_angle = slider_angle
        
        # Position data storage
        self.theta2_array = None
        self.x_B = None  # Crank pin position (joint B)
        self.y_B = None
        self.x_C = None  # Slider position (joint C)
        self.y_C = None
        self.theta3_array = None  # Connecting rod angle (joint angle)
        
        # Animation and plotting objects
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.animation = None
        
        # Trajectory data for animation
        self.traj_x = []
        self.traj_y = []
        
        # Inversion data cache
        self._inv_data = None
        
        # Check mechanism feasibility on init
        self._check_feasibility()
    
    def _check_feasibility(self) -> None:
        """Check if mechanism configuration is physically feasible"""
        print(f"\n{'='*50}")
        print("Crank-Slider Mechanism Analysis")
        print(f"{'='*50}")
        print(f"Crank length (Crank_length): {self.Crank_length}")
        print(f"Connecting rod length (rod_length): {self.rod_length}")
        print(f"Offset: {self.offset}")
        
        if self.offset == 0:
            print("Type: In-Line Slider-Crank")
            print("  - Balanced motion (symmetric stroke)")
        else:
            print("Type: Offset Slider-Crank")
            print("  - Quick-return mechanism (asymmetric stroke)")
        
        if self.rod_length <= self.Crank_length + abs(self.offset):
            print(f"\n⚠ WARNING: Connecting rod too short!")
            print(f"  Minimum length required: {self.Crank_length + abs(self.offset):.3f}")
        else:
            print(f"✓ Mechanism is feasible")
            stroke = 2 * self.Crank_length
            print(f"  Stroke length: {stroke:.3f}")
        print(f"{'='*50}\n")
    
    def calculate_positions(
        self,
        rot_num: int = 3,
        increment: float = 0.05
    ) -> None:
        """
        Calculate joint positions of crank-slider mechanism
        
        Parameters
        ----------
        rot_num : int
            Number of full crank rotations to simulate
        increment : float
            Angle increment per step in radians
        """
        print("Starting position calculation...")
        
        self.theta2_array = np.arange(0, rot_num * 2 * pi, increment)
        self.theta2_array = np.append(self.theta2_array, rot_num * 2 * pi)
        
        n_points = len(self.theta2_array)
        
        self.x_B = np.zeros(n_points)
        self.y_B = np.zeros(n_points)
        self.x_C = np.zeros(n_points)
        self.y_C = np.zeros(n_points)
        self.theta3_array = np.zeros(n_points)
        
        for i, theta2 in enumerate(self.theta2_array):
            self.x_B[i] = self.x_crank + self.Crank_length * cos(theta2)
            self.y_B[i] = self.y_crank + self.Crank_length * sin(theta2)
            
            y_offset = self.Crank_length * sin(theta2) - self.offset
            discriminant = self.rod_length**2 - y_offset**2
            
            if discriminant >= 0:
                x_slider = self.Crank_length * cos(theta2) + sqrt(discriminant)
                
                self.x_C[i] = self.x_crank + x_slider * cos(self.slider_angle) - \
                             self.offset * sin(self.slider_angle)
                self.y_C[i] = self.y_crank + x_slider * sin(self.slider_angle) + \
                             self.offset * cos(self.slider_angle)
                
                dx = self.x_C[i] - self.x_B[i]
                dy = self.y_C[i] - self.y_B[i]
                self.theta3_array[i] = arctan2(dy, dx)
            else:
                if i > 0:
                    self.x_C[i] = self.x_C[i-1]
                    self.y_C[i] = self.y_C[i-1]
                    self.theta3_array[i] = self.theta3_array[i-1]
        
        print(f"✓ Calculation complete! (Total {n_points} points)")
    
    def get_joint_angles(
        self, 
        in_degrees: bool = True,
        return_flat: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get joint angles and positions for original configuration
        
        Parameters
        ----------
        in_degrees : bool
            Return angles in degrees (default True)
        return_flat : bool
            Return flat structure for DataFrame compatibility
        
        Returns
        -------
        dict with keys:
            'time', 'crank_angle', 'rod_angle', 'slider_position',
            and joint positions 'joint_A', 'joint_B', 'joint_C' (either tuple or flat)
        """
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        # Slider displacement relative to initial
        slider_pos = np.sqrt((self.x_C - self.x_C[0])**2 + (self.y_C - self.y_C[0])**2)
        slider_pos[self.x_C < self.x_C[0]] *= -1
        
        x_A = np.full(len(self.theta2_array), self.x_crank)
        y_A = np.full(len(self.theta2_array), self.y_crank)
        
        if in_degrees:
            time_arr = np.degrees(self.theta2_array)
            crank_arr = np.degrees(self.theta2_array)
            rod_arr = np.degrees(self.theta3_array)
        else:
            time_arr = self.theta2_array
            crank_arr = self.theta2_array
            rod_arr = self.theta3_array
        
        if return_flat:
            return {
                'time': time_arr,
                'crank_angle': crank_arr,
                'rod_angle': rod_arr,
                'slider_position': slider_pos,
                'joint_A_x': x_A, 'joint_A_y': y_A,
                'joint_B_x': self.x_B, 'joint_B_y': self.y_B,
                'joint_C_x': self.x_C, 'joint_C_y': self.y_C
            }
        else:
            return {
                'time': time_arr,
                'crank_angle': crank_arr,
                'rod_angle': rod_arr,
                'slider_position': slider_pos,
                'joint_A': (x_A, y_A),
                'joint_B': (self.x_B, self.y_B),
                'joint_C': (self.x_C, self.y_C)
            }
    
    def calculate_inversion_angles(
        self,
        inversion_type: Literal[1, 2, 3, 4],
        in_degrees: bool = True,
        return_flat: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Calculate joint angles and positions for a specific inversion
        
        inversion_type:
            1 - Original (ground fixed)
            2 - Crank fixed
            3 - Connecting rod fixed
            4 - Slider fixed
        
        Returns dict with similar keys as get_joint_angles()
        """
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        if inversion_type == 1:
            return self.get_joint_angles(in_degrees=in_degrees, return_flat=return_flat)
        
        n_points = len(self.theta2_array)
        
        x_A_tf = np.zeros(n_points)
        y_A_tf = np.zeros(n_points)
        x_B_tf = np.zeros(n_points)
        y_B_tf = np.zeros(n_points)
        x_C_tf = np.zeros(n_points)
        y_C_tf = np.zeros(n_points)
        
        theta_crank = np.zeros(n_points)
        theta_rod = np.zeros(n_points)
        slider_pos = np.zeros(n_points)
        
        for i in range(n_points):
            if inversion_type == 2:
                angle = -self.theta2_array[i]
                ca = cos(angle); sa = sin(angle)
                
                x_A_tf[i] = 0; y_A_tf[i] = 0
                x_B_tf[i] = self.Crank_length; y_B_tf[i] = 0
                
                x_temp = self.x_C[i] - self.x_crank
                y_temp = self.y_C[i] - self.y_crank
                
                x_C_tf[i] = x_temp * ca - y_temp * sa
                y_C_tf[i] = x_temp * sa + y_temp * ca
                
                theta_crank[i] = 0
                theta_rod[i] = arctan2(y_C_tf[i]-y_B_tf[i], x_C_tf[i]-x_B_tf[i])
                slider_pos[i] = x_C_tf[i]
            
            elif inversion_type == 3:
                mid_x = (self.x_B[i] + self.x_C[i]) / 2
                mid_y = (self.y_B[i] + self.y_C[i]) / 2
                
                angle = -self.theta3_array[i]
                ca = cos(angle); sa = sin(angle)
                
                xA_temp = self.x_crank - mid_x
                yA_temp = self.y_crank - mid_y
                xB_temp = self.x_B[i] - mid_x
                yB_temp = self.y_B[i] - mid_y
                xC_temp = self.x_C[i] - mid_x
                yC_temp = self.y_C[i] - mid_y
                
                x_A_tf[i] = xA_temp * ca - yA_temp * sa
                y_A_tf[i] = xA_temp * sa + yA_temp * ca
                x_B_tf[i] = xB_temp * ca - yB_temp * sa
                y_B_tf[i] = xB_temp * sa + yB_temp * ca
                x_C_tf[i] = xC_temp * ca - yC_temp * sa
                y_C_tf[i] = xC_temp * sa + yC_temp * ca
                
                theta_crank[i] = arctan2(y_B_tf[i]-y_A_tf[i], x_B_tf[i]-x_A_tf[i])
                theta_rod[i] = 0
                slider_pos[i] = x_C_tf[i]
            
            elif inversion_type == 4:
                x_A_tf[i] = self.x_crank - self.x_C[i]
                y_A_tf[i] = self.y_crank - self.y_C[i]
                x_B_tf[i] = self.x_B[i] - self.x_C[i]
                y_B_tf[i] = self.y_B[i] - self.y_C[i]
                x_C_tf[i] = 0
                y_C_tf[i] = 0
                
                theta_crank[i] = arctan2(y_B_tf[i]-y_A_tf[i], x_B_tf[i]-x_A_tf[i])
                theta_rod[i] = arctan2(y_C_tf[i]-y_B_tf[i], x_C_tf[i]-x_B_tf[i])
                slider_pos[i] = 0
        
        frame_pos = np.sqrt(x_A_tf**2 + y_A_tf**2)
        
        if in_degrees:
            time_arr = np.degrees(self.theta2_array)
            crank_arr = np.degrees(theta_crank)
            rod_arr = np.degrees(theta_rod)
        else:
            time_arr = self.theta2_array
            crank_arr = theta_crank
            rod_arr = theta_rod
        
        if return_flat:
            return {
                'time': time_arr,
                'crank_angle': crank_arr,
                'rod_angle': rod_arr,
                'slider_position': slider_pos,
                'frame_position': frame_pos,
                'joint_A_x': x_A_tf, 'joint_A_y': y_A_tf,
                'joint_B_x': x_B_tf, 'joint_B_y': y_B_tf,
                'joint_C_x': x_C_tf, 'joint_C_y': y_C_tf
            }
        else:
            return {
                'time': time_arr,
                'crank_angle': crank_arr,
                'rod_angle': rod_arr,
                'slider_position': slider_pos,
                'frame_position': frame_pos,
                'joint_A': (x_A_tf, y_A_tf),
                'joint_B': (x_B_tf, y_B_tf),
                'joint_C': (x_C_tf, y_C_tf)
            }
    
    def get_inversion_angles_at_time(
        self,
        inversion_type: Literal[1, 2, 3, 4],
        time_index: int,
        in_degrees: bool = True
    ) -> Dict[str, float]:
        """
        Return joint angles and positions at given time index
        
        Parameters
        ----------
        inversion_type : int
        time_index : int
        in_degrees : bool
        
        Returns
        -------
        dict of angles and positions at that time step
        """
        all_angles = self.calculate_inversion_angles(inversion_type, in_degrees, return_flat=False)
        
        return {
            'time': all_angles['time'][time_index],
            'crank_angle': all_angles['crank_angle'][time_index],
            'rod_angle': all_angles['rod_angle'][time_index],
            'slider_position': all_angles['slider_position'][time_index],
            'frame_position': all_angles['frame_position'][time_index] if inversion_type == 4 else None,
            'joint_A': (all_angles['joint_A'][0][time_index], all_angles['joint_A'][1][time_index]),
            'joint_B': (all_angles['joint_B'][0][time_index], all_angles['joint_B'][1][time_index]),
            'joint_C': (all_angles['joint_C'][0][time_index], all_angles['joint_C'][1][time_index])
        }
    
    # 아래 animate 등 기존 메서드는 이전에 작성한 것과 같습니다.


    
    
    def print_inversion_summary(
        self,
        inversion_type: Literal[1, 2, 3, 4]
    ) -> None:
        """
        Print summary of inversion angles and ranges
        
        Parameters
        ----------
        inversion_type : int
            Inversion type (1, 2, 3, or 4)
        """
        angles = self.calculate_inversion_angles(inversion_type, in_degrees=True)
        
        inversion_names = {
            1: "Ground Fixed (Original)",
            2: "Crank Fixed",
            3: "Connecting Rod Fixed",
            4: "Slider Fixed"
        }
        
        print(f"\n{'='*60}")
        print(f"Inversion {inversion_type}: {inversion_names[inversion_type]}")
        print(f"{'='*60}")
        
        print(f"\nCrank Angle:")
        print(f"  Range: {angles['crank_angle'].min():.2f}° to {angles['crank_angle'].max():.2f}°")
        print(f"  Mean: {angles['crank_angle'].mean():.2f}°")
        print(f"  Std Dev: {angles['crank_angle'].std():.2f}°")
        
        print(f"\nConnecting Rod Angle:")
        print(f"  Range: {angles['rod_angle'].min():.2f}° to {angles['rod_angle'].max():.2f}°")
        print(f"  Mean: {angles['rod_angle'].mean():.2f}°")
        print(f"  Std Dev: {angles['rod_angle'].std():.2f}°")
        
        print(f"\nSlider Position:")
        print(f"  Range: {angles['slider_position'].min():.4f} m to {angles['slider_position'].max():.4f} m")
        print(f"  Stroke: {angles['slider_position'].max() - angles['slider_position'].min():.4f} m")
        
        if inversion_type == 4:
            print(f"\nFrame Position:")
            print(f"  Range: {angles['frame_position'].min():.4f} m to {angles['frame_position'].max():.4f} m")
        
        print(f"{'='*60}\n")

    def animate(
        self,
        interval: int = 30,
        trace_point: str = 'slider',
        show_slider_path: bool = True,
        figsize: Tuple[int, int] = (14, 6)
    ) -> None:
        """Create and display crank-slider animation"""
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        self.traj_x = []
        self.traj_y = []
        self.trace_point = trace_point
        self.show_slider_path = show_slider_path
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize)
        
        self._setup_mechanism_plot()
        self._setup_analysis_plot()
        
        print("Creating animation...")
        self.animation = animation.FuncAnimation(
            self.fig,
            self._animate_frame,
            init_func=self._init_animation,
            frames=len(self.theta2_array),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, wspace=0.3)
        plt.show()
    
    def _setup_mechanism_plot(self) -> None:
        """Setup mechanism plot"""
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        
        max_x = max(np.max(self.x_C), np.max(self.x_B)) + 1
        min_x = min(np.min(self.x_C), np.min(self.x_B), self.x_crank) - 1
        max_y = max(np.max(self.y_C), np.max(self.y_B)) + 1
        min_y = min(np.min(self.y_C), np.min(self.y_B), self.y_crank) - 1
        
        self.ax1.set_xlim(min_x, max_x)
        self.ax1.set_ylim(min_y, max_y)
        
        self.ax1.set_xlabel('X [m]', fontsize=12)
        self.ax1.set_ylabel('Y [m]', fontsize=12)
        self.ax1.set_title('Crank-Slider Mechanism', fontsize=14, fontweight='bold')
        
        self.ax1.plot(self.x_crank, self.y_crank, 'ko', markersize=12, zorder=5)
        self.ax1.text(self.x_crank-0.3, self.y_crank-0.4, 'A', 
                     fontsize=12, fontweight='bold')
        
        if self.show_slider_path:
            x_min_slider = np.min(self.x_C)
            x_max_slider = np.max(self.x_C)
            y_slider = self.y_crank + self.offset
            
            path_extend = 0.5
            x_path_start = x_min_slider - path_extend
            x_path_end = x_max_slider + path_extend
            
            if abs(self.slider_angle) < 0.01:
                self.ax1.plot([x_path_start, x_path_end], 
                            [y_slider, y_slider],
                            'k--', lw=2, alpha=0.3, label='Slider Path')
        
        self.crank_line, = self.ax1.plot(
            [], [], 'o-', lw=4, color='#d62728',
            markersize=8, label='Crank'
        )
        
        self.rod_line, = self.ax1.plot(
            [], [], 'o-', lw=4, color='#2b8cbe',
            markersize=8, label='Connecting Rod'
        )
        
        self.slider_patch = plt.Rectangle((0, 0), 0.3, 0.2, 
                                         fc='orange', ec='black', lw=2)
        self.ax1.add_patch(self.slider_patch)
        
        self.trajectory, = self.ax1.plot(
            [], [], '-', lw=2, color='green', alpha=0.7
        )
        
        self.ax1.legend(fontsize=10, loc='upper right')
    
    def _setup_analysis_plot(self) -> None:
        """Setup analysis plot"""
        slider_displacement = np.sqrt((self.x_C - self.x_C[0])**2 + 
                                     (self.y_C - self.y_C[0])**2)
        
        for i in range(len(slider_displacement)):
            if self.x_C[i] < self.x_C[0]:
                slider_displacement[i] *= -1
        
        ax2_twin = self.ax2.twinx()
        
        self.ax2.plot(
            np.degrees(self.theta2_array),
            slider_displacement,
            label='Slider Displacement', lw=2, color='blue'
        )
        
        ax2_twin.plot(
            np.degrees(self.theta2_array),
            np.degrees(self.theta3_array),
            label='Rod Angle', lw=2, color='red'
        )
        
        self.ax2.set_xlabel('Crank Angle [deg]', fontsize=12)
        self.ax2.set_ylabel('Slider Displacement [m]', fontsize=12, color='blue')
        ax2_twin.set_ylabel('Connecting Rod Angle [deg]', fontsize=12, color='red')
        self.ax2.set_title('Kinematic Analysis', fontsize=14, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper left', fontsize=10)
        ax2_twin.legend(loc='upper right', fontsize=10)
        
        self.current_point, = self.ax2.plot([], [], 'go', markersize=8)
    
    def _init_animation(self):
        """Initialize animation"""
        self.traj_x = []
        self.traj_y = []
        self.crank_line.set_data([], [])
        self.rod_line.set_data([], [])
        self.trajectory.set_data([], [])
        self.current_point.set_data([], [])
        self.slider_patch.set_xy((0, 0))
        return (self.crank_line, self.rod_line, self.trajectory, 
                self.current_point, self.slider_patch)
    
    def _animate_frame(self, frame: int):
        """Update animation frame"""
        self.crank_line.set_data(
            [self.x_crank, self.x_B[frame]],
            [self.y_crank, self.y_B[frame]]
        )
        
        self.rod_line.set_data(
            [self.x_B[frame], self.x_C[frame]],
            [self.y_B[frame], self.y_C[frame]]
        )
        
        slider_width = 0.4
        slider_height = 0.3
        slider_x = self.x_C[frame] - slider_width / 2
        slider_y = self.y_C[frame] - slider_height / 2
        self.slider_patch.set_xy((slider_x, slider_y))
        
        if self.trace_point == 'slider':
            traj_x, traj_y = self.x_C[frame], self.y_C[frame]
        elif self.trace_point == 'crank_pin':
            traj_x, traj_y = self.x_B[frame], self.y_B[frame]
        elif self.trace_point == 'rod_mid':
            traj_x = (self.x_B[frame] + self.x_C[frame]) / 2
            traj_y = (self.y_B[frame] + self.y_C[frame]) / 2
        else:
            traj_x, traj_y = self.x_C[frame], self.y_C[frame]
        
        self.traj_x.append(traj_x)
        self.traj_y.append(traj_y)
        
        points_per_cycle = int(2 * pi / (self.theta2_array[1] - self.theta2_array[0]))
        if len(self.traj_x) > points_per_cycle * 1.5:
            self.traj_x = self.traj_x[-int(points_per_cycle * 1.5):]
            self.traj_y = self.traj_y[-int(points_per_cycle * 1.5):]
        
        self.trajectory.set_data(self.traj_x, self.traj_y)
        
        self.current_point.set_data(
            [np.degrees(self.theta2_array[frame])],
            [sqrt((self.x_C[frame] - self.x_C[0])**2 + 
                  (self.y_C[frame] - self.y_C[0])**2)]
        )
        
        return (self.crank_line, self.rod_line, self.trajectory,
                self.current_point, self.slider_patch)
    
    def kinematic_inversion(
        self,
        inversion_type: Literal[1, 2, 3, 4],
        interval: int = 30,
        figsize: Tuple[int, int] = (16, 6)
    ) -> None:
        """Perform kinematic inversion of slider-crank mechanism"""
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        inversion_names = {
            1: "First Inversion - Ground Fixed (Reciprocating Engine)",
            2: "Second Inversion - Crank Fixed (Quick-Return Mechanism)",
            3: "Third Inversion - Connecting Rod Fixed (Oscillating Engine)",
            4: "Fourth Inversion - Slider Fixed (Hand Pump)"
        }
        
        print(f"\n{'='*60}")
        print(f"Kinematic Inversion: {inversion_names[inversion_type]}")
        print(f"{'='*60}\n")
        
        if inversion_type == 1:
            self.animate(interval=interval, figsize=figsize)
        else:
            self._animate_inversion(inversion_type, interval, figsize)
    
    def _animate_inversion(
        self,
        inversion_type: int,
        interval: int,
        figsize: Tuple[int, int]
    ) -> None:
        """Animate inversion configurations"""
        # Get angles using the new function
        angles_data = self.calculate_inversion_angles(inversion_type, in_degrees=False)
        
        # Store in _inv_data for animation
        self._inv_data = {
            'x_A': angles_data['joint_A'][0],
            'y_A': angles_data['joint_A'][1],
            'x_B': angles_data['joint_B'][0],
            'y_B': angles_data['joint_B'][1],
            'x_C': angles_data['joint_C'][0],
            'y_C': angles_data['joint_C'][1],
            'type': inversion_type,
            'theta_crank': angles_data['crank_angle'],
            'theta_rod': angles_data['rod_angle'],
            'slider_pos': angles_data['slider_position']
        }
        
        self.fig = plt.figure(figsize=figsize)
        self.ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        self.ax2 = plt.subplot2grid((2, 2), (0, 1))
        self.ax3 = plt.subplot2grid((2, 2), (1, 1))
        
        self._setup_inversion_plot(inversion_type)
        self._setup_inversion_angle_plots(inversion_type)
        
        self.traj_x = []
        self.traj_y = []
        
        self.animation = animation.FuncAnimation(
            self.fig,
            self._animate_inversion_frame,
            init_func=self._init_inversion_animation,
            frames=len(self.theta2_array),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        plt.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.08, 
                          hspace=0.35, wspace=0.35)
        plt.show()
    
    def _setup_inversion_plot(self, inversion_type: int) -> None:
        """Setup plot for inversion animation"""
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        
        inv_titles = {
            2: "Quick-Return Mechanism (Crank Fixed)",
            3: "Oscillating Cylinder Engine (Rod Fixed)",
            4: "Hand Pump (Slider Fixed)"
        }
        
        self.ax1.set_title(inv_titles[inversion_type], 
                          fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('X [m]', fontsize=12)
        self.ax1.set_ylabel('Y [m]', fontsize=12)
        
        inv = self._inv_data
        all_x = np.concatenate([inv['x_A'], inv['x_B'], inv['x_C']])
        all_y = np.concatenate([inv['y_A'], inv['y_B'], inv['y_C']])
        margin = 1.0
        self.ax1.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        self.ax1.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
        
        self.crank_line, = self.ax1.plot(
            [], [], 'o-', lw=4, color='#d62728', markersize=10,
            label='Crank (Fixed)' if inversion_type == 2 else 'Crank'
        )
        self.rod_line, = self.ax1.plot(
            [], [], 'o-', lw=4, color='#2b8cbe', markersize=10,
            label='Rod (Fixed)' if inversion_type == 3 else 'Connecting Rod'
        )
        self.trajectory, = self.ax1.plot(
            [], [], '-', lw=2, color='green', alpha=0.7, label='Trajectory'
        )
        
        if inversion_type == 2:
            self.ax1.plot(0, 0, 'ks', markersize=12, zorder=10, label='Fixed Pivot A')
            self.ax1.plot(self.Crank_length, 0, 'rs', markersize=12, zorder=10, label='Fixed Pin B')
        elif inversion_type == 4:
            self.ax1.plot(0, 0, 'ks', markersize=12, zorder=10, label='Fixed Slider')
        
        self.ax1.legend(fontsize=9, loc='best')
    
    def _setup_inversion_angle_plots(self, inversion_type: int) -> None:
        """Setup angle visualization plots"""
        inv = self._inv_data
        time_array = np.degrees(self.theta2_array)
        
        if inversion_type == 2:
            self.ax2.plot(time_array, np.degrees(inv['theta_rod']), 
                         'b-', lw=2, label='Rod Angle')
            self.ax2.set_ylabel('Rod Angle [deg]', fontsize=11)
            self.ax2.set_title('Angular DOF', fontsize=12, fontweight='bold')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend(fontsize=10)
            
        elif inversion_type == 3:
            self.ax2.plot(time_array, np.degrees(inv['theta_crank']),
                         'r-', lw=2, label='Crank Angle')
            self.ax2.set_ylabel('Crank Angle [deg]', fontsize=11)
            self.ax2.set_title('Angular DOF', fontsize=12, fontweight='bold')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend(fontsize=10)
            
        elif inversion_type == 4:
            self.ax2.plot(time_array, np.degrees(inv['theta_crank']),
                         'r-', lw=2, label='Crank Angle')
            self.ax2.plot(time_array, np.degrees(inv['theta_rod']),
                         'b-', lw=2, label='Rod Angle')
            self.ax2.set_ylabel('Angle [deg]', fontsize=11)
            self.ax2.set_title('Angular DOFs', fontsize=12, fontweight='bold')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend(fontsize=10)
        
        self.ax2.set_xlabel('Time (Crank Cycles) [deg]', fontsize=11)
        self.angle_marker, = self.ax2.plot([], [], 'ro', markersize=8)
        
        if inversion_type in [2, 3]:
            self.ax3.plot(time_array, inv['slider_pos'], 'g-', lw=2)
            self.ax3.set_xlabel('Time (Crank Cycles) [deg]', fontsize=11)
            self.ax3.set_ylabel('Slider Position [m]', fontsize=11)
            self.ax3.set_title('Linear DOF', fontsize=12, fontweight='bold')
            self.ax3.grid(True, alpha=0.3)
            self.slider_marker, = self.ax3.plot([], [], 'go', markersize=8)
        else:
            frame_x = inv['x_A']
            frame_y = inv['y_A']
            frame_pos = np.sqrt(frame_x**2 + frame_y**2)
            self.ax3.plot(time_array, frame_pos, 'm-', lw=2)
            self.ax3.set_xlabel('Time (Crank Cycles) [deg]', fontsize=11)
            self.ax3.set_ylabel('Frame Position [m]', fontsize=11)
            self.ax3.set_title('Frame Motion', fontsize=12, fontweight='bold')
            self.ax3.grid(True, alpha=0.3)
            self.slider_marker, = self.ax3.plot([], [], 'mo', markersize=8)
    
    def _init_inversion_animation(self):
        """Initialize inversion animation"""
        self.traj_x = []
        self.traj_y = []
        self.crank_line.set_data([], [])
        self.rod_line.set_data([], [])
        self.trajectory.set_data([], [])
        self.angle_marker.set_data([], [])
        self.slider_marker.set_data([], [])
        return (self.crank_line, self.rod_line, self.trajectory,
                self.angle_marker, self.slider_marker)
    
    def _animate_inversion_frame(self, frame: int):
        """Animate inversion frame"""
        inv = self._inv_data
        
        self.crank_line.set_data(
            [inv['x_A'][frame], inv['x_B'][frame]],
            [inv['y_A'][frame], inv['y_B'][frame]]
        )
        
        self.rod_line.set_data(
            [inv['x_B'][frame], inv['x_C'][frame]],
            [inv['y_B'][frame], inv['y_C'][frame]]
        )
        
        if inv['type'] == 2:
            traj_x, traj_y = inv['x_C'][frame], inv['y_C'][frame]
        elif inv['type'] == 3:
            traj_x, traj_y = inv['x_A'][frame], inv['y_A'][frame]
        else:
            traj_x, traj_y = inv['x_B'][frame], inv['y_B'][frame]
        
        self.traj_x.append(traj_x)
        self.traj_y.append(traj_y)
        
        points_per_cycle = int(2 * pi / (self.theta2_array[1] - self.theta2_array[0]))
        if len(self.traj_x) > points_per_cycle * 1.5:
            self.traj_x = self.traj_x[-int(points_per_cycle * 1.5):]
            self.traj_y = self.traj_y[-int(points_per_cycle * 1.5):]
        
        self.trajectory.set_data(self.traj_x, self.traj_y)
        
        time_val = np.degrees(self.theta2_array[frame])
        if inv['type'] == 2:
            angle_val = np.degrees(inv['theta_rod'][frame])
        elif inv['type'] == 3:
            angle_val = np.degrees(inv['theta_crank'][frame])
        else:
            angle_val = np.degrees(inv['theta_crank'][frame])
        
        self.angle_marker.set_data([time_val], [angle_val])
        self.slider_marker.set_data([time_val], [inv['slider_pos'][frame]])
        
        return (self.crank_line, self.rod_line, self.trajectory,
                self.angle_marker, self.slider_marker)
    
    def __repr__(self) -> str:
        """Class representation"""
        return (f"CrankSlider(Crank_length={self.Crank_length}, rod_length={self.rod_length}, "
                f"offset={self.offset})")



# 사용 예시
if __name__ == "__main__":
    mech = CrankSlider(Crank_length=2.0, rod_length=6.0, offset=0.0)
    mech.calculate_positions(rot_num=2, increment=0.05)
    
    # 전체 각도 배열 반환 (예: inversion 2)
    angles = mech.calculate_inversion_angles(2, in_degrees=True, return_flat=True)
    import pandas as pd
    df = pd.DataFrame(angles)
    print(df.head())
    
    # 특정 시간 단계에서의 각도 가져오기
    time_index = 50
    angle_at_time = mech.get_inversion_angles_at_time(2, time_index, in_degrees=True)
    print(f"Time {angle_at_time['time']:.2f} deg, Crank angle: {angle_at_time['crank_angle']:.2f}, Rod angle: {angle_at_time['rod_angle']:.2f}")
   # Create mechanism
    crank_slider = CrankSlider(Crank_length=2.0, rod_length=8.0, offset=0.0)
    crank_slider.calculate_positions(rot_num=2, increment=0.05)
    
    # Get original configuration angles
    print("\n" + "="*60)
    print("ORIGINAL CONFIGURATION ANGLES")
    print("="*60)
    angles_orig = crank_slider.get_joint_angles(in_degrees=True)
    print(f"Time range: {angles_orig['time'][0]:.1f}° to {angles_orig['time'][-1]:.1f}°")
    print(f"Crank angle range: {angles_orig['crank_angle'].min():.1f}° to {angles_orig['crank_angle'].max():.1f}°")
    print(f"Rod angle range: {angles_orig['rod_angle'].min():.1f}° to {angles_orig['rod_angle'].max():.1f}°")
    print(f"Slider stroke: {angles_orig['slider_position'].max() - angles_orig['slider_position'].min():.4f} m")
    
    # Get inversion 2 angles
    print("\n" + "="*60)
    print("INVERSION 2 (CRANK FIXED) ANGLES")
    print("="*60)
    angles_inv2 = crank_slider.calculate_inversion_angles(2, in_degrees=True)
    print(f"Crank angle (should be 0): {angles_inv2['crank_angle'].min():.1f}° to {angles_inv2['crank_angle'].max():.1f}°")
    print(f"Rod angle range: {angles_inv2['rod_angle'].min():.1f}° to {angles_inv2['rod_angle'].max():.1f}°")
    print(f"Slider position range: {angles_inv2['slider_position'].min():.4f} m to {angles_inv2['slider_position'].max():.4f} m")
    
    # Print detailed summary for all inversions
    for i in range(1, 5):
        crank_slider.print_inversion_summary(i)
    
    # Get angles at specific time
    print("\n" + "="*60)
    print("ANGLES AT SPECIFIC TIME (frame 50)")
    print("="*60)
    angles_at_50 = crank_slider.get_inversion_angles_at_time(2, 50, in_degrees=True)
    print(f"Time: {angles_at_50['time']:.2f}°")
    print(f"Crank angle: {angles_at_50['crank_angle']:.2f}°")
    print(f"Rod angle: {angles_at_50['rod_angle']:.2f}°")
    print(f"Slider position: {angles_at_50['slider_position']:.4f} m")
    print(f"Joint B position: ({angles_at_50['joint_B'][0]:.4f}, {angles_at_50['joint_B'][1]:.4f})")
    
    # Visualize
    # crank_slider.animate(interval=30)
    # 1 : motor fixed
    # 2 : crank fixed
    # 3 : rod fixed
    # 4 : slider fixed
    crank_slider.kinematic_inversion(2, interval=30)
