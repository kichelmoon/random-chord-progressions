# Harmonic Algebra according to Neo-Riemannian theory
class HarmonicAlgebra:
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    @staticmethod
    def triad(root: int, is_major: bool = True) -> tuple[int, int, int]:
        third = 4 if is_major else 3
        return (root % 12, (root + third) % 12, (root + 7) % 12)

    # Parallel - Switching minor and major, preserves perfect fifth. So P(P(x)) = x
    @classmethod
    def P(cls, triad: tuple[int, int, int]) -> tuple[int, int, int]:
        root, third, fifth = triad
        is_major = (third - root) % 12 == 4
        new_third = (third - 1) % 12 if is_major else (third + 1) % 12
        return (root, new_third, fifth)

    # Relative - preserves major third interval
    @classmethod
    def R(cls, triad: tuple[int, int, int]) -> tuple[int, int, int]:
        root, third, fifth = triad
        is_major = (third - root) % 12 == 4
        if is_major:
            return ((root - 2) % 12, third, root)
        else:
            return (fifth, (fifth + 4) % 12, (fifth + 1) % 12)

    # Leading tone exchange - preserves minor third
    @classmethod
    def L(cls, triad: tuple[int, int, int]) -> tuple[int, int, int]:
        root, third, fifth = triad
        is_major = (third - root) % 12 == 4
        if is_major:
            return (third, fifth, (root - 1) % 12)
        else:
            return ((fifth + 1) % 12, root, third)