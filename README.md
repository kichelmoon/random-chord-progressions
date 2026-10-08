# Random Chord Progressions

Using Neo-Riemannian Theory, Differential Geometry and Markov Chains to generate random sequences of chords and cool visualizations for them

## Previewing the Harmonic Manifold

In Neo-Riemannian Theory, we take a look at the orbits of three group actions: Parallel, Relative and leading tone. This naturally leads to a manifold structure that is diffeomorphic to a torus wich we can picture in three dimensions.\
On the torus, we can do basic moves from differential geometry: Look at flows and geodesics. We can look at riemannian metrics to calculate distances that respect the curvature of the torus.\
All of this has music theoretic meaning: Shorter path correspond to chord changes that minimize finger movement playing a piano. The file DemoHarmonicManifold.py creates some of these pictures. They are pretty nice to look at, even without understanding the musical implications.
![Manifold Demo][demos/demo_1_geodesic_flow.png]

## Previewing the Harmonic Markov Chain

Using the Harmonic Manifold with our new notion of distance, we can create a markov chain that gives rise to a random walk on the torus (meaning a random chord progression). Each chord has three notes, we calculate the distance for each and use a Boltzmann like softmax distribution. Note that we disable self-loops, staying at the same chord is not an option for us.\
The file DemoHarmonicMarkovChain.py shows one instance of the random walk. An interesting feature is the compactness of the torus: A random walk in euclidian 3-space usually drifts away from the starting point relatively quickly, here we stay close to our tonal centre where we started.
![Markov Chain Demo][demos/demo_4_random_walk.png]
