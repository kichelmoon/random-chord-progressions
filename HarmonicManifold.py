import numpy as np

class HarmonicManifold:
    def __init__(self, R: float = 2.5, r: float = 1.0):
        self.R = R  # Major radius (Circle of Fifths)
        self.r = r  # Minor radius (Chromatic Circle)

    def pitch_to_angles(self, p: int) -> tuple[float, float]:
        """Maps pitch class p to (theta1: Chromatic, theta2: Circle of Fifths)."""
        theta1 = 2 * np.pi * (p % 12) / 12.0
        theta2 = 2 * np.pi * ((7 * p) % 12) / 12.0
        return theta1, theta2

    def angles_to_torus(self, theta1: float, theta2: float) -> np.ndarray:
        """Embeds angle pair (theta1, theta2) into R^3."""
        x = (self.R + self.r * np.cos(theta1)) * np.cos(theta2)
        y = (self.R + self.r * np.cos(theta1)) * np.sin(theta2)
        z = self.r * np.sin(theta1)
        return np.array([x, y, z])

    def embed_pitch(self, p: int) -> np.ndarray:
        t1, t2 = self.pitch_to_angles(p)
        return self.angles_to_torus(t1, t2)

    def geodesic_arc(self, t1_a: float, t2_a: float, t1_b: float, t2_b: float, num_samples: int = 50) -> np.ndarray:
        """Generates dense sampling along the intrinsic surface geodesic path."""
        def shortest_signed_angle_diff(a, b):
            return (b - a + np.pi) % (2 * np.pi) - np.pi

        dt1 = shortest_signed_angle_diff(t1_a, t1_b)
        dt2 = shortest_signed_angle_diff(t2_a, t2_b)

        t_vals = np.linspace(0, 1, num_samples)
        arc = []
        for t in t_vals:
            t1_curr = t1_a + t * dt1
            t2_curr = t2_a + t * dt2
            arc.append(self.angles_to_torus(t1_curr, t2_curr))
        return np.array(arc)