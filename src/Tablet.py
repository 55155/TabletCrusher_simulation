import pylife.strength.sn_curve

def tablet_hardness(hardness, diameter, thickness):
    """
    Calculate tablet hardness based on tensile strength using USP formula.

    Parameters:
    -----------
    tensile_strength : float
        Tensile strength of the tablet material (MPa or N/mm^2).
    diameter : float
        Diameter of the tablet (mm).
    thickness : float
        Thickness of the tablet (mm).

    Returns:
    --------
    hardness : float
        Calculated hardness force (in Newtons, N).
    """
    import math

    # USP Tensile Strength formula for flat-faced tablets:
    # Tensile Strength (σ) = 2 * F / (π * D * t)
    # => F (hardness force) = σ * π * D * t / 2
    tensile_strength = 2 * hardness / (math.pi * diameter * thickness)
    return tensile_strength
# dummy data

# not convex 
Tablet_tensile_strength = tablet_hardness(hardness=360, diameter=10, thickness=5)
Tablet_Fatigue_life = pylife.strength.sn_curve.FiniteLifeBase(-.1, .4*Tablet_tensile_strength, 1e3)
print(Tablet_Fatigue_life)