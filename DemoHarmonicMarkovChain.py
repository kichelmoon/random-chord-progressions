import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import matplotlib.cm as cm

from HarmonicAlgebra import HarmonicAlgebra
from HarmonicManifold import HarmonicManifold
from HarmonicMarkovChain import HarmonicMarkovChain
 
background_color = '#0F172A'
mesh_color = '#F1F5F9'
accent_color = '#FFB800'

chords = []
chord_labels = []
for root in range(12):
    chords.append(HarmonicAlgebra.triad(root, is_major=True))
    chord_labels.append(f"{HarmonicAlgebra.NOTE_NAMES[root]}")
    chords.append(HarmonicAlgebra.triad(root, is_major=False))
    chord_labels.append(f"{HarmonicAlgebra.NOTE_NAMES[root]}m")

manifold = HarmonicManifold(R=2.5, r=1.0)
markov = HarmonicMarkovChain(manifold, chords, beta=0.55)

seq_indices = markov.generate_sequence(start_idx=0, steps=12, seed=101)
gen_chords = [chords[i] for i in seq_indices]
gen_labels = [chord_labels[i] for i in seq_indices]

start_chord = gen_chords[0]
distances_from_start = [manifold.chord_geodesic(start_chord, c) for c in gen_chords]
step_to_step_dists = [manifold.chord_geodesic(gen_chords[i], gen_chords[i+1]) for i in range(len(gen_chords)-1)]

fig1 = plt.figure(figsize=(11, 9), facecolor=background_color)
ax1 = fig1.add_subplot(111, projection='3d', facecolor=background_color)

u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
U, V = np.meshgrid(u, v)
X = (manifold.R + manifold.r * np.cos(U)) * np.cos(V)
Y = (manifold.R + manifold.r * np.cos(U)) * np.sin(V)
Z = manifold.r * np.sin(U)
ax1.plot_wireframe(X, Y, Z, color=mesh_color, alpha=0.15, linewidth=0.5)

cmap = cm.cool
colors = cmap(np.linspace(0, 1, len(gen_chords) - 1))

for i in range(len(gen_chords) - 1):
    arc = manifold.geodesic_arc(gen_chords[i], gen_chords[i+1])
    ax1.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=colors[i], linewidth=3.5, alpha=0.9)

centroids = np.array([manifold.chord_centroid(c) for c in gen_chords])
ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='#ffffff', s=60, zorder=6)

# Label Start and End Chords
start_pt = centroids[0]
end_pt = centroids[-1]
ax1.scatter(start_pt[0], start_pt[1], start_pt[2], color='#00ffcc', s=140, zorder=10, label=f"Start: {gen_labels[0]}")
ax1.scatter(end_pt[0], end_pt[1], end_pt[2], color='#ff007f', s=140, zorder=10, label=f"End: {gen_labels[-1]}")

for idx, (pt, label) in enumerate(zip(centroids, gen_labels)):
    ax1.text(pt[0]*1.06, pt[1]*1.06, pt[2]*1.06 + 0.1, f"{idx+1}.{label}", color='#ecf0f1', fontsize=9, fontweight='bold')

ax1.set_title("Figure 1: Generated Markov Chain Trajectory on $\mathbb{T}^2$", color='white', fontsize=13, pad=15)
ax1.set_axis_off()
ax1.legend(loc='lower right', facecolor='#161b26', edgecolor='none', labelcolor='white')
plt.tight_layout()
plt.savefig("demos/demo_4_random_walk.png", dpi=300, facecolor=fig1.get_facecolor())
plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0b0f19')
ax2.set_facecolor('#111625')

steps_range = np.arange(1, len(gen_chords) + 1)

# Main Distance Line
ax2.plot(steps_range, distances_from_start, color='#00e5ff', marker='o', linewidth=2.5, markersize=8, label=r"$d(C_k, C_1)$ to Start Chord")
ax2.fill_between(steps_range, distances_from_start, color='#00e5ff', alpha=0.12)

# Annotate each point with chord label
for idx, (s, d, label) in enumerate(zip(steps_range, distances_from_start, gen_labels)):
    ax2.annotate(f"{label}\n({d:.2f})", (s, d), textcoords="offset points", xytext=(0, 12),
                 ha='center', color='white', fontsize=9, fontweight='bold')

ax2.set_xticks(steps_range)
ax2.set_xticklabels([f"Step {s}" for s in steps_range], color='#a0aec0', fontsize=10)
ax2.tick_params(colors='#a0aec0')
ax2.grid(True, linestyle='--', alpha=0.2, color='#4a5568')

ax2.set_xlabel("Progression Steps", color='white', fontsize=11, labelpad=10)
ax2.set_ylabel(r"Geodesic Distance $d(C_k, C_1)$", color='white', fontsize=11)
ax2.set_title(f"Figure 2: Harmonic Drift from Starting Chord ({gen_labels[0]})", color='white', fontsize=13, pad=15)
ax2.set_ylim(0, max(distances_from_start) * 1.35)

for spine in ax2.spines.values():
    spine.set_color('#2d3748')

plt.tight_layout()
plt.savefig("demos/demo_5_random_walk_distance_traveled.png", dpi=300, facecolor=fig2.get_facecolor())
plt.show()