import numpy as np
import plotly.graph_objects as go

def slider_crank_kinematics(r, L, theta_rad):
    # Position (inline slider-crank)
    under = L**2 - (r*np.sin(theta_rad))**2
    if np.any(under < 0):
        raise ValueError("Invalid geometry: need L >= r (and numerical margin).")
    x = r*np.cos(theta_rad) + np.sqrt(under)

    # dx/dtheta (theta in radians)
    dx_dtheta = (
        -r*np.sin(theta_rad)
        - (r**2 * np.sin(theta_rad) * np.cos(theta_rad)) / np.sqrt(under)
    )
    return x, dx_dtheta

def quasi_static_curves(r, L, n_pts=2001, eps=1e-9):
    theta_deg = np.linspace(0.0, 360.0, n_pts)
    theta = np.deg2rad(theta_deg)

    x, dx_dtheta = slider_crank_kinematics(r, L, theta)

    # Avoid singularities near dead centers where dx/dtheta -> 0
    mask = np.abs(dx_dtheta) < eps
    dx_dtheta_safe = dx_dtheta.copy()
    dx_dtheta_safe[mask] = np.nan

    # Normalized reference load/torque
    F_ref = 1.0   # N (slider axial force)
    T_ref = 1.0   # N·m (crank torque)

    # From virtual work: T = -F * dx/dtheta  (theta in rad)
    T_from_unit_F = -F_ref * dx_dtheta_safe          # N·m, for F=1N
    F_from_unit_T = -T_ref / dx_dtheta_safe          # N,   for T=1N·m

    return theta_deg, x, dx_dtheta, T_from_unit_F, F_from_unit_T

def plot_curves(theta_deg, T_curve, F_curve):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=theta_deg, y=T_curve, mode="lines", name="Torque (for F=1 N)"))
    fig1.update_layout(
        title="Quasi-static Torque vs Crank Angle (normalized: F_slider = 1 N)",
        xaxis_title="Crank angle (degree)",
        yaxis_title="Torque (N·m)",
        template="plotly_white",
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=theta_deg, y=F_curve, mode="lines", name="Axial force (for T=1 N·m)"))
    fig2.update_layout(
        title="Quasi-static Slider Axial Force vs Crank Angle (normalized: T = 1 N·m)",
        xaxis_title="Crank angle (degree)",
        yaxis_title="Axial force Fx (N)",
        template="plotly_white",
    )
    return fig1, fig2

if __name__ == "__main__":
    # ===== User inputs (meters) =====
    r = 0.02   # crank radius [m]
    L = 0.08   # connecting rod length [m]
    # ================================

    theta_deg, x, dx_dtheta, T_curve, F_curve = quasi_static_curves(r, L)

    figT, figF = plot_curves(theta_deg, T_curve, F_curve)
    figT.show()
    figF.show()

    # Scaling tips:
    # - If your actual slider axial force is F_actual [N], then Torque = (F_actual/1N) * T_curve
    # - If your actual torque is T_actual [N·m], then Axial force = (T_actual/1N·m) * F_curve
