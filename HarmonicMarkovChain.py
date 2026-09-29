import numpy as np
from HarmonicManifold import HarmonicManifold

class HarmonicMarkovChain:
    def __init__(self, manifold: HarmonicManifold, chord_space: list[tuple[int, ...]], beta: float = 0.5):
        self.manifold = manifold
        self.chord_space = chord_space
        self.beta = beta
        self.P = self._build_transition_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        n = len(self.chord_space)
        P = np.zeros((n, n))
        for i, c1 in enumerate(self.chord_space):
            for j, c2 in enumerate(self.chord_space):
                if i != j:
                    d = self.manifold.chord_geodesic(c1, c2)
                    P[i, j] = np.exp(-self.beta * d)
            row_sum = np.sum(P[i, :])
            if row_sum > 0:
                P[i, :] /= row_sum
        return P

    def generate_sequence(self, start_idx: int, steps: int = 12, seed: int = 42) -> list[int]:
        np.random.seed(seed)
        current = start_idx
        path = [current]
        for _ in range(steps - 1):
            current = np.random.choice(len(self.chord_space), p=self.P[current, :])
            path.append(current)
        return path