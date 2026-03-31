import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import pi, sin, cos, sqrt, arctan2, arccos
from typing import Tuple, Optional, Literal


class FourBarLinkage:
    """
    Four-bar Linkage Mechanism Class
    
    Parameters
    ----------
    r1 : float
        Ground link length
    r2 : float
        Crank link length
    r3 : float
        Coupler link length
    r4 : float
        Rocker link length
    ground_position : tuple, optional
        Start and end coordinates of ground link ((x_A, y_A), (x_D, y_D))
    """
    
    def __init__(
        self, 
        r1: float, 
        r2: float, 
        r3: float, 
        r4: float,
        ground_position: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    ):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        
        # Set ground link position
        if ground_position is None:
            self.x_A, self.y_A = 0, 0
            self.x_D, self.y_D = r1, 0
        else:
            (self.x_A, self.y_A), (self.x_D, self.y_D) = ground_position
        
        # Position data storage variables
        self.theta2_array = None
        self.theta3_array = None
        self.theta4_array = None
        self.x_B = None
        self.y_B = None
        self.x_C = None
        self.y_C = None
        
        # Animation objects
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.animation = None
        self.traj_x = []
        self.traj_y = []
        
        # Check Grashof condition
        self._check_grashof_condition()
    
    def _check_grashof_condition(self) -> None:
        """Check Grashof condition and print mechanism type"""
        links = [self.r1, self.r2, self.r3, self.r4]
        s = min(links)
        l = max(links)
        p_q_sum = sum(links) - s - l
        
        print(f"\n{'='*50}")
        print("Four-Bar Linkage Mechanism Analysis")
        print(f"{'='*50}")
        print(f"Link lengths: r1={self.r1}, r2={self.r2}, r3={self.r3}, r4={self.r4}")
        print(f"S={s}, L={l}, P+Q={p_q_sum:.2f}")
        
        if s + l < p_q_sum:
            print("✓ Grashof Mechanism: At least one link can rotate 360 degrees")
            
            if self.r1 == s:
                print("  Type: Double Crank")
            elif self.r2 == s or self.r4 == s:
                print("  Type: Crank-Rocker")
            else:
                print("  Type: Double Rocker")
        elif s + l > p_q_sum:
            print("✗ Non-Grashof Mechanism: Triple Rocker")
        else:
            print("◯ Change Point: S + L = P + Q")
        print(f"{'='*50}\n")
    
    def calculate_positions(
        self, 
        rot_num: int = 3, 
        increment: float = 0.05
    ) -> None:
        """
        Calculate joint positions of four-bar linkage
        
        Parameters
        ----------
        rot_num : int
            Number of rotations
        increment : float
            Angle increment (radians)
        """
        print("Starting position calculation...")
        
        # Generate crank angle array
        self.theta2_array = np.arange(0, rot_num * 2 * pi, increment)
        self.theta2_array = np.append(self.theta2_array, rot_num * 2 * pi)
        
        n_points = len(self.theta2_array)
        
        # Initialize arrays
        self.x_B = np.zeros(n_points)
        self.y_B = np.zeros(n_points)
        self.x_C = np.zeros(n_points)
        self.y_C = np.zeros(n_points)
        self.theta3_array = np.zeros(n_points)
        self.theta4_array = np.zeros(n_points)
        
        # Calculate positions for each frame
        for i, theta2 in enumerate(self.theta2_array):
            # Joint B position (crank endpoint)
            self.x_B[i] = self.x_A + self.r2 * cos(theta2)
            self.y_B[i] = self.y_A + self.r2 * sin(theta2)
            
            # Calculate theta4 (vector loop equation)
            K1 = self.r1 / self.r2
            K2 = self.r1 / self.r4
            K3 = (self.r2**2 - self.r3**2 + self.r4**2 + self.r1**2) / (2 * self.r2 * self.r4)
            
            A = cos(theta2) - K1 - K2 * cos(theta2) + K3
            B = -2 * sin(theta2)
            C = K1 - (K2 + 1) * cos(theta2) + K3
            
            discriminant = B**2 - 4*A*C
            
            if discriminant >= 0:
                theta4 = 2 * arctan2(-B + sqrt(discriminant), 2*A)
            else:
                theta4 = self.theta4_array[i-1] if i > 0 else 0
            
            self.theta4_array[i] = theta4
            
            # Calculate theta3
            K4 = self.r1 / self.r3
            K5 = (self.r4**2 - self.r1**2 - self.r2**2 - self.r3**2) / (2 * self.r2 * self.r3)
            
            D = cos(theta2) - K1 + K4 * cos(theta2) + K5
            E = -2 * sin(theta2)
            F = K1 + (K4 - 1) * cos(theta2) + K5
            
            discriminant2 = E**2 - 4*D*F
            
            if discriminant2 >= 0:
                theta3 = 2 * arctan2(-E + sqrt(discriminant2), 2*D)
            else:
                theta3 = self.theta3_array[i-1] if i > 0 else 0
            
            self.theta3_array[i] = theta3
            
            # Joint C position (coupler endpoint)
            self.x_C[i] = self.x_B[i] + self.r3 * cos(theta3)
            self.y_C[i] = self.y_B[i] + self.r3 * sin(theta3)
        
        print(f"✓ Calculation complete! (Total {n_points} frames)")
    
    def kinematic_inversion(
        self,
        fixed_joint_1: Literal['A', 'B', 'C', 'D'],
        fixed_joint_2: Literal['A', 'B', 'C', 'D'],
        interval: int = 30,
        figsize: Tuple[int, int] = (14, 6)
    ) -> None:
        """
        Perform kinematic inversion by fixing different links as ground
        
        Parameters
        ----------
        fixed_joint_1 : str
            First fixed joint ('A', 'B', 'C', 'D')
        fixed_joint_2 : str
            Second fixed joint ('A', 'B', 'C', 'D')
        interval : int
            Animation interval in milliseconds
        figsize : tuple
            Figure size
            
        Examples
        --------
        Original configuration (A-D fixed):
            linkage.kinematic_inversion('A', 'D')
        
        First inversion (A-B fixed - Link 2 as ground):
            linkage.kinematic_inversion('A', 'B')
        
        Second inversion (B-C fixed - Link 3 as ground):
            linkage.kinematic_inversion('B', 'C')
        
        Third inversion (C-D fixed - Link 4 as ground):
            linkage.kinematic_inversion('C', 'D')
        """
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        # Validate inputs
        valid_joints = ['A', 'B', 'C', 'D']
        if fixed_joint_1 not in valid_joints or fixed_joint_2 not in valid_joints:
            raise ValueError(f"Joints must be one of {valid_joints}")
        
        if fixed_joint_1 == fixed_joint_2:
            raise ValueError("Two different joints must be specified")
        
        # Determine inversion type
        fixed_pair = ''.join(sorted([fixed_joint_1, fixed_joint_2]))
        inversion_map = {
            'AD': ('Link 1 (Ground)', 'Original Configuration', self.r1),
            'AB': ('Link 2 (Crank)', 'First Inversion', self.r2),
            'BC': ('Link 3 (Coupler)', 'Second Inversion', self.r3),
            'CD': ('Link 4 (Rocker)', 'Third Inversion', self.r4)
        }
        
        if fixed_pair not in inversion_map:
            raise ValueError(f"Invalid joint pair. Valid pairs: AD, AB, BC, CD")
        
        link_name, inversion_name, fixed_length = inversion_map[fixed_pair]
        
        print(f"\n{'='*60}")
        print(f"Kinematic Inversion: {inversion_name}")
        print(f"Fixed Link: {fixed_joint_1}-{fixed_joint_2} ({link_name})")
        print(f"Fixed Link Length: {fixed_length:.3f}")
        print(f"{'='*60}\n")
        
        # Get transformed coordinates
        x_A_tf, y_A_tf, x_B_tf, y_B_tf, x_C_tf, y_C_tf, x_D_tf, y_D_tf = \
            self._transform_to_inversion(fixed_joint_1, fixed_joint_2)
        
        # Create figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Setup plots with transformed coordinates
        self._setup_inversion_mechanism_plot(
            x_A_tf, y_A_tf, x_D_tf, y_D_tf,
            fixed_joint_1, fixed_joint_2, inversion_name
        )
        self._setup_angle_plot()
        
        # Store transformed data
        self._inv_data = {
            'x_A': x_A_tf, 'y_A': y_A_tf,
            'x_B': x_B_tf, 'y_B': y_B_tf,
            'x_C': x_C_tf, 'y_C': y_C_tf,
            'x_D': x_D_tf, 'y_D': y_D_tf,
            'fixed_1': fixed_joint_1,
            'fixed_2': fixed_joint_2
        }
        
        # Initialize trajectory
        self.traj_x = []
        self.traj_y = []
        
        # Create animation
        print("Creating inversion animation...")
        self.animation = animation.FuncAnimation(
            self.fig,
            self._animate_inversion_frame,
            init_func=self._init_animation,
            frames=len(self.theta2_array),
            interval=interval,
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
    
    def _transform_to_inversion(
        self,
        fixed_joint_1: str,
        fixed_joint_2: str
    ) -> Tuple[np.ndarray, ...]:
        """
        Transform coordinates for kinematic inversion
        
        Returns
        -------
        x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D : arrays
            Transformed coordinates for all joints
        """
        n_points = len(self.theta2_array)
        
        # Get original joint coordinates for each frame
        # These are already calculated
        x_A_orig = np.full(n_points, self.x_A)
        y_A_orig = np.full(n_points, self.y_A)
        x_B_orig = self.x_B
        y_B_orig = self.y_B
        x_C_orig = self.x_C
        y_C_orig = self.y_C
        x_D_orig = np.full(n_points, self.x_D)
        y_D_orig = np.full(n_points, self.y_D)
        
        # Create a mapping
        joint_coords = {
            'A': (x_A_orig, y_A_orig),
            'B': (x_B_orig, y_B_orig),
            'C': (x_C_orig, y_C_orig),
            'D': (x_D_orig, y_D_orig)
        }
        
        # Get coordinates of the two fixed joints
        x1, y1 = joint_coords[fixed_joint_1]
        x2, y2 = joint_coords[fixed_joint_2]
        
        # Transform: place first joint at origin, second on positive x-axis
        x_A_tf = np.zeros(n_points)
        y_A_tf = np.zeros(n_points)
        x_B_tf = np.zeros(n_points)
        y_B_tf = np.zeros(n_points)
        x_C_tf = np.zeros(n_points)
        y_C_tf = np.zeros(n_points)
        x_D_tf = np.zeros(n_points)
        y_D_tf = np.zeros(n_points)
        
        for i in range(n_points):
            # Translation: move first fixed joint to origin
            dx = x1[i]
            dy = y1[i]
            
            # Rotation: align second fixed joint to x-axis
            angle = arctan2(y2[i] - y1[i], x2[i] - x1[i])
            
            # Apply transformation to all joints
            for joint_name in ['A', 'B', 'C', 'D']:
                x_orig, y_orig = joint_coords[joint_name]
                
                # Translate
                x_t = x_orig[i] - dx
                y_t = y_orig[i] - dy
                
                # Rotate
                x_r = x_t * cos(-angle) - y_t * sin(-angle)
                y_r = x_t * sin(-angle) + y_t * cos(-angle)
                
                # Store
                if joint_name == 'A':
                    x_A_tf[i], y_A_tf[i] = x_r, y_r
                elif joint_name == 'B':
                    x_B_tf[i], y_B_tf[i] = x_r, y_r
                elif joint_name == 'C':
                    x_C_tf[i], y_C_tf[i] = x_r, y_r
                elif joint_name == 'D':
                    x_D_tf[i], y_D_tf[i] = x_r, y_r
        
        return x_A_tf, y_A_tf, x_B_tf, y_B_tf, x_C_tf, y_C_tf, x_D_tf, y_D_tf
    
    def _setup_inversion_mechanism_plot(
        self,
        x_fixed_1: np.ndarray,
        y_fixed_1: np.ndarray,
        x_fixed_2: np.ndarray,
        y_fixed_2: np.ndarray,
        joint_1_name: str,
        joint_2_name: str,
        inversion_name: str
    ) -> None:
        """Setup mechanism plot for inversion"""
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        
        # Auto axis range
        max_reach = max(self.r1, self.r2, self.r3, self.r4) * 2 + 1
        self.ax1.set_xlim(-max_reach, max_reach)
        self.ax1.set_ylim(-max_reach, max_reach)
        
        self.ax1.set_xlabel('X [m]', fontsize=12)
        self.ax1.set_ylabel('Y [m]', fontsize=12)
        self.ax1.set_title(f'Four-Bar Linkage - {inversion_name}', 
                          fontsize=14, fontweight='bold')
        
        # Initialize link line
        self.line, = self.ax1.plot(
            [], [], 'o-', lw=4, color='#2b8cbe',
            markersize=8, markerfacecolor='red'
        )
        
        # Initialize trajectory line
        self.trajectory, = self.ax1.plot(
            [], [], '-', lw=2, color='orange', alpha=0.8
        )
        
        # Show fixed joints
        self.ax1.plot(x_fixed_1[0], y_fixed_1[0], 'ko', markersize=12, zorder=5)
        self.ax1.plot(x_fixed_2[0], y_fixed_2[0], 'ko', markersize=12, zorder=5)
        self.ax1.text(
            x_fixed_1[0]-0.3, y_fixed_1[0]-0.5, joint_1_name,
            fontsize=12, fontweight='bold'
        )
        self.ax1.text(
            x_fixed_2[0]+0.2, y_fixed_2[0]-0.5, joint_2_name,
            fontsize=12, fontweight='bold'
        )
        
        # Add fixed link indicator
        fixed_link_length = sqrt((x_fixed_2[0] - x_fixed_1[0])**2 + 
                                (y_fixed_2[0] - y_fixed_1[0])**2)
        self.ax1.plot([x_fixed_1[0], x_fixed_2[0]], 
                     [y_fixed_1[0], y_fixed_2[0]], 
                     'k-', lw=6, alpha=0.3, zorder=1,
                     label=f'Fixed Link ({joint_1_name}-{joint_2_name})')
        self.ax1.legend(fontsize=10, loc='upper right')
    
    def _animate_inversion_frame(self, frame: int):
        """Update animation frame for inversion"""
        inv = self._inv_data
        
        # Draw linkage in correct order
        x_points = [inv['x_A'][frame], inv['x_B'][frame], 
                   inv['x_C'][frame], inv['x_D'][frame]]
        y_points = [inv['y_A'][frame], inv['y_B'][frame], 
                   inv['y_C'][frame], inv['y_D'][frame]]
        
        self.line.set_data(x_points, y_points)
        
        # Trace moving joint (not one of the fixed joints)
        moving_joints = set(['A', 'B', 'C', 'D']) - set([inv['fixed_1'], inv['fixed_2']])
        moving_joint = list(moving_joints)[0]  # Pick first moving joint
        
        if moving_joint == 'A':
            traj_x, traj_y = inv['x_A'][frame], inv['y_A'][frame]
        elif moving_joint == 'B':
            traj_x, traj_y = inv['x_B'][frame], inv['y_B'][frame]
        elif moving_joint == 'C':
            traj_x, traj_y = inv['x_C'][frame], inv['y_C'][frame]
        else:  # 'D'
            traj_x, traj_y = inv['x_D'][frame], inv['y_D'][frame]
        
        self.traj_x.append(traj_x)
        self.traj_y.append(traj_y)
        
        # Limit trajectory length
        points_per_cycle = int(2 * pi / (self.theta2_array[1] - self.theta2_array[0]))
        if len(self.traj_x) > points_per_cycle * 1.5:
            self.traj_x = self.traj_x[-int(points_per_cycle * 1.5):]
            self.traj_y = self.traj_y[-int(points_per_cycle * 1.5):]
        
        self.trajectory.set_data(self.traj_x, self.traj_y)
        
        # Update angle graph
        self.current_point.set_data(
            [np.degrees(self.theta2_array[frame])],
            [np.degrees(self.theta3_array[frame])]
        )
        
        return self.line, self.trajectory, self.current_point
    
    def get_coupler_point(
        self, 
        ratio: float = 0.5, 
        offset: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return coordinates of a specific point on the coupler link
        
        Parameters
        ----------
        ratio : float
            Ratio on coupler link (0: point B, 1: point C)
        offset : float
            Perpendicular offset from the link
            
        Returns
        -------
        x, y : numpy arrays
            x, y coordinates of the point
        """
        x = self.x_B + ratio * (self.x_C - self.x_B)
        y = self.y_B + ratio * (self.y_C - self.y_B)
        
        if offset != 0:
            # Add perpendicular offset
            theta = self.theta3_array
            x += offset * cos(theta + pi/2)
            y += offset * sin(theta + pi/2)
        
        return x, y
    
    def animate(
        self, 
        interval: int = 30, 
        trace_point: str = 'coupler_mid',
        show_full_trajectory: bool = False,
        figsize: Tuple[int, int] = (14, 6)
    ) -> None:
        """
        Create and display four-bar linkage animation
        
        Parameters
        ----------
        interval : int
            Time interval between frames (milliseconds)
        trace_point : str
            Point to trace trajectory ('coupler_mid', 'joint_B', 'joint_C')
        show_full_trajectory : bool
            Whether to show full trajectory in advance (fade effect)
        figsize : tuple
            Figure size
        """
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        # Initialize trajectory data
        self.traj_x = []
        self.traj_y = []
        self.trace_point = trace_point
        self.show_full_trajectory = show_full_trajectory
        
        # Create figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left subplot: linkage mechanism
        self._setup_mechanism_plot()
        
        # Right subplot: angle graph
        self._setup_angle_plot()
        
        # Show full trajectory in advance (option)
        if show_full_trajectory:
            self._draw_full_trajectory()
        
        # Create animation
        print("Creating animation...")
        self.animation = animation.FuncAnimation(
            self.fig, 
            self._animate_frame, 
            init_func=self._init_animation,
            frames=len(self.theta2_array), 
            interval=interval, 
            blit=True, 
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
    
    def _setup_mechanism_plot(self) -> None:
        """Setup mechanism plot"""
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        
        # Auto axis range setting
        max_reach = self.r2 + self.r3
        x_center = self.r1 / 2
        self.ax1.set_xlim(x_center - max_reach - 1, x_center + max_reach + 1)
        self.ax1.set_ylim(-max_reach - 1, max_reach + 1)
        
        self.ax1.set_xlabel('X [m]', fontsize=12)
        self.ax1.set_ylabel('Y [m]', fontsize=12)
        self.ax1.set_title('Four-Bar Linkage Mechanism', fontsize=14, fontweight='bold')
        
        # Initialize link line
        self.line, = self.ax1.plot(
            [], [], 'o-', lw=4, color='#2b8cbe', 
            markersize=8, markerfacecolor='red'
        )
        
        # Initialize trajectory line
        self.trajectory, = self.ax1.plot(
            [], [], '-', lw=2, color='orange', alpha=0.8
        )
        
        # Show fixed joints
        self.ax1.plot(self.x_A, self.y_A, 'ko', markersize=12, zorder=5)
        self.ax1.plot(self.x_D, self.y_D, 'ko', markersize=12, zorder=5)
        self.ax1.text(
            self.x_A-0.3, self.y_A-0.4, 'A', 
            fontsize=12, fontweight='bold'
        )
        self.ax1.text(
            self.x_D+0.2, self.y_D-0.4, 'D', 
            fontsize=12, fontweight='bold'
        )
    
    def _draw_full_trajectory(self) -> None:
        """Show full trajectory in semi-transparent"""
        # Extract data for one cycle only
        points_per_cycle = int(2 * pi / (self.theta2_array[1] - self.theta2_array[0]))
        
        if self.trace_point == 'coupler_mid':
            x_full = (self.x_B[:points_per_cycle] + self.x_C[:points_per_cycle]) / 2
            y_full = (self.y_B[:points_per_cycle] + self.y_C[:points_per_cycle]) / 2
        elif self.trace_point == 'joint_B':
            x_full = self.x_B[:points_per_cycle]
            y_full = self.y_B[:points_per_cycle]
        elif self.trace_point == 'joint_C':
            x_full = self.x_C[:points_per_cycle]
            y_full = self.y_C[:points_per_cycle]
        else:
            x_full = (self.x_B[:points_per_cycle] + self.x_C[:points_per_cycle]) / 2
            y_full = (self.y_B[:points_per_cycle] + self.y_C[:points_per_cycle]) / 2
        
        self.ax1.plot(x_full, y_full, '--', lw=1, color='gray', alpha=0.3, zorder=1)
    
    def _setup_angle_plot(self) -> None:
        """Setup angle plot"""
        self.ax2.plot(
            np.degrees(self.theta2_array), 
            np.degrees(self.theta3_array), 
            label='Theta3 (Coupler)', lw=2, color='green'
        )
        self.ax2.plot(
            np.degrees(self.theta2_array), 
            np.degrees(self.theta4_array), 
            label='Theta4 (Rocker)', lw=2, color='purple'
        )
        self.ax2.set_xlabel('Theta2 (Crank Angle) [deg]', fontsize=12)
        self.ax2.set_ylabel('Angle [deg]', fontsize=12)
        self.ax2.set_title('Link Angle Variation', fontsize=14, fontweight='bold')
        self.ax2.legend(fontsize=11)
        self.ax2.grid(True, alpha=0.3)
        
        # Current position marker
        self.current_point, = self.ax2.plot(
            [], [], 'ro', markersize=8
        )
    
    def _init_animation(self):
        """Initialize animation"""
        self.traj_x = []
        self.traj_y = []
        self.line.set_data([], [])
        self.trajectory.set_data([], [])
        self.current_point.set_data([], [])
        return self.line, self.trajectory, self.current_point
    
    def _animate_frame(self, frame: int):
        """Update animation frame"""
        # Draw links
        x_points = [self.x_A, self.x_B[frame], self.x_C[frame], self.x_D]
        y_points = [self.y_A, self.y_B[frame], self.y_C[frame], self.y_D]
        self.line.set_data(x_points, y_points)
        
        # Calculate point to trace
        if self.trace_point == 'coupler_mid':
            traj_x = (self.x_B[frame] + self.x_C[frame]) / 2
            traj_y = (self.y_B[frame] + self.y_C[frame]) / 2
        elif self.trace_point == 'joint_B':
            traj_x = self.x_B[frame]
            traj_y = self.y_B[frame]
        elif self.trace_point == 'joint_C':
            traj_x = self.x_C[frame]
            traj_y = self.y_C[frame]
        else:
            traj_x = (self.x_B[frame] + self.x_C[frame]) / 2
            traj_y = (self.y_B[frame] + self.y_C[frame]) / 2
        
        self.traj_x.append(traj_x)
        self.traj_y.append(traj_y)
        
        # Limit trajectory to recent 1.5 cycles
        points_per_cycle = int(2 * pi / (self.theta2_array[1] - self.theta2_array[0]))
        if len(self.traj_x) > points_per_cycle * 1.5:
            self.traj_x = self.traj_x[-int(points_per_cycle * 1.5):]
            self.traj_y = self.traj_y[-int(points_per_cycle * 1.5):]
        
        self.trajectory.set_data(self.traj_x, self.traj_y)
        
        # Current position on angle graph
        self.current_point.set_data(
            [np.degrees(self.theta2_array[frame])], 
            [np.degrees(self.theta3_array[frame])]
        )
        
        return self.line, self.trajectory, self.current_point
    
    def save_animation(
        self, 
        filename: str, 
        fps: int = 30, 
        writer: str = 'pillow'
    ) -> None:
        """
        Save animation to file
        
        Parameters
        ----------
        filename : str
            File name to save (e.g., 'linkage.gif', 'linkage.mp4')
        fps : int
            Frames per second
        writer : str
            Save library ('pillow', 'ffmpeg', 'imagemagick')
        """
        if self.animation is None:
            raise ValueError("Please run animate() first!")
        
        print(f"Saving animation: {filename}")
        self.animation.save(filename, writer=writer, fps=fps)
        print(f"✓ Save complete: {filename}")
    
    def get_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate angular velocities (numerical differentiation)
        
        Returns
        -------
        omega3, omega4 : numpy arrays
            Angular velocities of coupler and rocker
        """
        omega3 = np.gradient(self.theta3_array, self.theta2_array)
        omega4 = np.gradient(self.theta4_array, self.theta2_array)
        return omega3, omega4
    
    def plot_trajectory(
        self, 
        point_ratio: float = 0.5, 
        offset: float = 0.0,
        num_cycles: int = 1
    ) -> None:
        """
        Plot trajectory of a specific point on coupler link
        
        Parameters
        ----------
        point_ratio : float
            Ratio on coupler link
        offset : float
            Perpendicular offset from link
        num_cycles : int
            Number of cycles to display (1: one cycle only, 0: all)
        """
        if self.theta2_array is None:
            raise ValueError("Please run calculate_positions() first!")
        
        x, y = self.get_coupler_point(point_ratio, offset)
        
        # Calculate points per cycle
        points_per_cycle = int(2 * pi / (self.theta2_array[1] - self.theta2_array[0]))
        
        if num_cycles == 0:
            # Display all
            x_plot = x
            y_plot = y
        else:
            # Display specified number of cycles
            end_idx = min(points_per_cycle * num_cycles + 1, len(x))
            x_plot = x[:end_idx]
            y_plot = y[:end_idx]
        
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # Plot trajectory
        ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, 
                label=f'Coupler Curve (ratio={point_ratio}, offset={offset})')
        
        # Show start and end points
        ax.plot(x_plot[0], y_plot[0], 'go', markersize=12, 
                label='Start Point', zorder=5)
        
        if num_cycles == 1 or (num_cycles > 0 and len(x_plot) >= points_per_cycle):
            ax.plot(x_plot[-1], y_plot[-1], 'ro', markersize=12, 
                    label='End Point (1 cycle)', zorder=5)
        
        # Show fixed joints
        ax.plot(self.x_A, self.y_A, 'ks', markersize=14, 
                label='Fixed Joint A', zorder=5)
        ax.plot(self.x_D, self.y_D, 'ks', markersize=14, 
                label='Fixed Joint D', zorder=5)
        
        ax.set_xlabel('X [m]', fontsize=13)
        ax.set_ylabel('Y [m]', fontsize=13)
        ax.set_title(f'Coupler Curve - {num_cycles if num_cycles > 0 else "All"} Cycle(s)', 
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')
        ax.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """Class representation"""
        return (f"FourBarLinkage(r1={self.r1}, r2={self.r2}, "
                f"r3={self.r3}, r4={self.r4})")


# ========== Usage Example ==========
if __name__ == "__main__":
    # Create four-bar linkage
    linkage = FourBarLinkage(
        r1=4.0,  # Ground link
        r2=1.0,  # Crank
        r3=3.5,  # Coupler
        r4=3.0   # Rocker
    )
    
    # Calculate positions (one cycle)
    linkage.calculate_positions(rot_num=1, increment=0.05)
    
    # Original configuration (A-D fixed)
    print("\n=== Original Configuration ===")
    linkage.animate(interval=30, trace_point='coupler_mid', show_full_trajectory=True)
    
    # First Inversion: Fix A-B (Link 2 as ground)
    print("\n=== First Inversion: A-B Fixed ===")
    linkage.kinematic_inversion('A', 'B', interval=30)
    
    # Second Inversion: Fix B-C (Link 3 as ground)
    print("\n=== Second Inversion: B-C Fixed ===")
    linkage.kinematic_inversion('B', 'C', interval=30)
    
    # Third Inversion: Fix C-D (Link 4 as ground)
    print("\n=== Third Inversion: C-D Fixed ===")
    linkage.kinematic_inversion('C', 'D', interval=30)
