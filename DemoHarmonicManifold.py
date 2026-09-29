import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from HarmonicAlgebra import HarmonicAlgebra
from HarmonicManifold import HarmonicManifold

background_color = '#0F172A'
mesh_color = '#F1F5F9'
accent_color = '#FFB800'

all_chords = []
all_labels = []
for root in range(12):
    all_chords.append(HarmonicAlgebra.triad(root, is_major=True))
    all_labels.append(f"{HarmonicAlgebra.NOTE_NAMES[root]}")
    all_chords.append(HarmonicAlgebra.triad(root, is_major=False))
    all_labels.append(f"{HarmonicAlgebra.NOTE_NAMES[root]}m")


def plot_mesh_torus(manifold, ax):
    u_dense = np.linspace(0, 2 * np.pi, 40)
    v_dense = np.linspace(0, 2 * np.pi, 40)
    Ud, Vd = np.meshgrid(u_dense, v_dense)
    Xd = (manifold.R + manifold.r * np.cos(Ud)) * np.cos(Vd)
    Yd = (manifold.R + manifold.r * np.cos(Ud)) * np.sin(Vd)
    Zd = manifold.r * np.sin(Ud)
    ax.plot_wireframe(Xd, Yd, Zd, color=mesh_color, alpha=0.15, linewidth=0.5)
    
    
def demo_geodesic_flow(manifold: HarmonicManifold):
    c_major = HarmonicAlgebra.triad(0, is_major=True)
    chord_distances = []
    for chord, label in zip(all_chords, all_labels):
        if chord == c_major:
            continue
        dist = manifold.chord_geodesic(c_major, chord)
        chord_distances.append((dist, chord, label))

    chord_distances.sort(key=lambda x: x[0])
    closest_5 = chord_distances[:5]
    
    fig = plt.figure(figsize=(12, 10), facecolor=background_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=background_color)

    # Plot Torus Mesh
    plot_mesh_torus(manifold, ax)

    c_center = manifold.chord_centroid(c_major)
    ax.scatter(c_center[0], c_center[1], c_center[2], color='#00ffcc', s=160, zorder=10, label="C-Major (Origin)")
    ax.text(c_center[0]*1.08, c_center[1]*1.08, c_center[2] + 0.15, "C-Major", color='#00ffcc', fontsize=12, fontweight='bold')

    colors = cm.spring(np.linspace(0.1, 0.9, 5))

    for idx, (dist, chord, label) in enumerate(closest_5):
        arc = manifold.geodesic_arc(c_major, chord)
    
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=colors[idx], linewidth=3.0, alpha=0.9, label=f"{label} (d={dist:.2f})")
    
        target_pt = arc[-1]
        ax.scatter(target_pt[0], target_pt[1], target_pt[2], color=colors[idx], s=90, depthshade=False, zorder=8)
        ax.text(target_pt[0]*1.06, target_pt[1]*1.06, target_pt[2] + 0.1, f"{label}", color='white', fontsize=10, fontweight='bold')

    ax.set_title("Demo 1: 5 Closest Chords to C-Major on $\mathbb{T}^2$ via geodesics", color='white', fontsize=14, pad=20)
    ax.set_axis_off()
    ax.legend(loc='lower right', facecolor='#161b26', edgecolor='none', labelcolor='white', fontsize=10)

    plt.tight_layout()
    plt.savefig("demos/demo_1_geodesic_flow.png", dpi=300, facecolor=fig.get_facecolor())
    plt.show()


def demo_surface_distance_heatmap(manifold: HarmonicManifold, reference_pitch: int = 0):
    """
    Renders a continuous geodesic distance heatmap mapped directly onto the 
    surface of the torus, showing harmonic proximity relative to C.
    """
    fig = plt.figure(figsize=(12, 10), facecolor=background_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=background_color)

    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, 2 * np.pi, 120)
    U, V = np.meshgrid(u, v)

    ref_t1, ref_t2 = manifold.pitch_to_angles(reference_pitch)

    def angular_dist(a, b):
        diff = np.abs(a - b) % (2 * np.pi)
        return np.minimum(diff, 2 * np.pi - diff)

    D1 = angular_dist(U, ref_t1)
    D2 = angular_dist(V, ref_t2)
    Dist = np.sqrt((manifold.r * D1)**2 + (manifold.R * D2)**2)
    Dist_norm = (Dist - Dist.min()) / (Dist.max() - Dist.min())

    X = (manifold.R + manifold.r * np.cos(U)) * np.cos(V)
    Y = (manifold.R + manifold.r * np.cos(U)) * np.sin(V)
    Z = manifold.r * np.sin(U)

    surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis_r(Dist_norm), 
                           rstride=1, cstride=1, antialiased=True, alpha=0.88)

    # Highlicht the Pitch Classes
    for p in range(12):
        pt = manifold.embed_pitch(p)
        color = '#00ffcc' if p == reference_pitch else '#ffffff'
        ax.scatter(pt[0], pt[1], pt[2], color=color, s=80, depthshade=False, zorder=10)
        ax.text(pt[0]*1.12, pt[1]*1.12, pt[2]*1.12, f" {HarmonicAlgebra.NOTE_NAMES[p]}",
                color=color, fontsize=11, fontweight='bold')

    ax.set_title(f"Demo 2: Geodesic Metric Heatmap centered on Note '{HarmonicAlgebra.NOTE_NAMES[reference_pitch]}'", 
                 color='white', fontsize=14, pad=20)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("demos/demo_2_surface_heatmap.png", dpi=300, facecolor=fig.get_facecolor())
    plt.show()


def demo_vector_field_flow(manifold: HarmonicManifold):
    """
    Renders a 3D tangent vector field on the torus pointing along the gradient 
    of harmonic attraction toward the C-Major triad.
    """
    fig = plt.figure(figsize=(12, 10), facecolor=background_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=background_color)

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, 2 * np.pi, 24)
    U, V = np.meshgrid(u, v)

    target_t1 = np.mean([manifold.pitch_to_angles(p)[0] for p in (0, 4, 7)])
    target_t2 = np.mean([manifold.pitch_to_angles(p)[1] for p in (0, 4, 7)])

    dU = (target_t1 - U + np.pi) % (2 * np.pi) - np.pi
    dV = (target_t2 - V + np.pi) % (2 * np.pi) - np.pi

    X = (manifold.R + manifold.r * np.cos(U)) * np.cos(V)
    Y = (manifold.R + manifold.r * np.cos(U)) * np.sin(V)
    Z = manifold.r * np.sin(U)

    # Use Jacobian: (d/d_theta1, d/d_theta2) -> (dx, dy, dz)
    dX = -manifold.r * np.sin(U) * np.cos(V) * dU - (manifold.R + manifold.r * np.cos(U)) * np.sin(V) * dV
    dY = -manifold.r * np.sin(U) * np.sin(V) * dU + (manifold.R + manifold.r * np.cos(U)) * np.cos(V) * dV
    dZ = manifold.r * np.cos(U) * dU

    norm = np.sqrt(dX**2 + dY**2 + dZ**2)
    norm[norm == 0] = 1.0
    dX /= norm
    dY /= norm
    dZ /= norm

    ax.quiver(X, Y, Z, dX, dY, dZ, length=0.25, color=accent_color, alpha=0.75, linewidth=1.2)

    plot_mesh_torus(manifold, ax)

    ax.set_title("Demo 3: Tangent Vector Field Flow toward C-Major", 
                 color='white', fontsize=14, pad=20)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("demos/demo_3_vector_field.png", dpi=300, facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    manifold = HarmonicManifold(R=2.5, r=1.0)
    
    demo_geodesic_flow(manifold)
    demo_surface_distance_heatmap(manifold, reference_pitch=0)
    demo_vector_field_flow(manifold)
