import numpy as np
from itertools import permutations

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

    def chord_centroid_angles(self, chord: tuple[int, ...]) -> tuple[float, float]:
        angles = [self.pitch_to_angles(p) for p in chord]
        sin_t1 = np.mean([np.sin(t1) for t1, _ in angles])
        cos_t1 = np.mean([np.cos(t1) for t1, _ in angles])
        sin_t2 = np.mean([np.sin(t2) for _, t2 in angles])
        cos_t2 = np.mean([np.cos(t2) for _, t2 in angles])
        return np.arctan2(sin_t1, cos_t1), np.arctan2(sin_t2, cos_t2)

    def chord_centroid(self, chord: tuple[int, ...]) -> np.ndarray:
        t1, t2 = self.chord_centroid_angles(chord)
        return self.angles_to_torus(t1, t2)

    def pitch_geodesic(self, p1: int, p2: int) -> float:
        t1_a, t2_a = self.pitch_to_angles(p1)
        t1_b, t2_b = self.pitch_to_angles(p2)
        
        def angular_dist(a, b):
            diff = np.abs(a - b) % (2 * np.pi)
            return np.minimum(diff, 2 * np.pi - diff)

        d1 = angular_dist(t1_a, t1_b)
        d2 = angular_dist(t2_a, t2_b)
        return np.sqrt((self.r * d1)**2 + (self.R * d2)**2)

    def chord_geodesic(self, chord_A: tuple[int, ...], chord_B: tuple[int, ...]) -> float:
        """Minimal Voice Leading Distance (Infimum over permutation group S_3)."""
        min_dist = float('inf')
        for perm in permutations(chord_B):
            dist = np.sqrt(sum(self.pitch_geodesic(a, b)**2 for a, b in zip(chord_A, perm)))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def geodesic_arc(self, chord_A: tuple[int, ...], chord_B: tuple[int, ...], num_samples: int = 80) -> np.ndarray:
        t1_a, t2_a = self.chord_centroid_angles(chord_A)
        t1_b, t2_b = self.chord_centroid_angles(chord_B)

        def shortest_signed_diff(a, b):
            return (b - a + np.pi) % (2 * np.pi) - np.pi

        dt1 = shortest_signed_diff(t1_a, t1_b)
        dt2 = shortest_signed_diff(t2_a, t2_b)

        t_vals = np.linspace(0, 1, num_samples)
        arc = [self.angles_to_torus(t1_a + t * dt1, t2_a + t * dt2) for t in t_vals]
        return np.array(arc)