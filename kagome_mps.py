"""
kagome_mps.py
=============
Scaling the HVA past the ~24-qubit statevector wall, using
`AerSimulator(method='matrix_product_state')` (MPS).

Contents:
  - kagome_star_chain(k)      : generates the "star chain" lattice family
                                (the SAME construction as the advisor's 19-site
                                lattice: hexagonal cells of 6 triangles that share a
                                pair of adjacent triangles). Sizes: 12, 19, 26, 33,
                                40, 47, ... (+7 sites and +4 triangles per cell).
                                With k=2 the graph is isomorphic to the advisor's
                                19-site one (verified).
  - dimer_cover(...)          : dimer cover via maximum matching (networkx).
  - MPSEnergyEvaluator        : SYMMETRIC HVA circuit (θ shared per edge class,
                                few parameters) evaluated with Aer's MPS estimator.
  - run_mps_sweep_symmetric   : progressive layer sweep with warm start.

Energy references WITHOUT exact diagonalization (impossible at these sizes):
  - lower bound  E >= -3*T   (T = number of triangles; each triangle contributes
                              >= -3 in the ×4 convention — the same heuristic
                              argument the paper uses for the 103-site system),
  - baseline     E_dimer = -3*M  (M = number of dimers; the reps=0 state, exact),
  - per site     E/(4N) to compare with the thermodynamic limit (~-0.4386 J).

The adjoint gradient is NOT available through Aer, and the full HVA has too many
parameters for gradient-free optimization: hence at this scale we use the SYMMETRIC
variant (~5-6 params/layer), exactly the motivation laid out in Secs. 4.3/4.4.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

from kagome_hva import heisenberg_gate, heisenberg_hamiltonian, edge_color_classes


# --------------------------------------------------------------------------
# 1. Lattice generator: chain of "Stars of David"
# --------------------------------------------------------------------------
def kagome_star_chain(n_cells):
    """
    Chain of `n_cells` hexagonal Kagome cells. Each cell = a ring of 6 corner-sharing
    triangles; consecutive cells share a pair of adjacent triangles (the corner that
    joins them is the graph's "chord"). With n_cells=2 the construction reproduces
    (up to relabeling) the advisor's 19-site lattice.

    Returns (n_sites, edges, triangles):
      edges     : list of pairs [i,j] (each edge belongs to exactly 1 triangle)
      triangles : list of triples (a,b,c)
    Sites are numbered in creation order along the chain, which yields a 1D-local
    labeling favorable for MPS simulation.
    """
    assert n_cells >= 1
    nxt = 0
    def new():
        nonlocal nxt
        nxt += 1
        return nxt - 1

    triangles = []   # each triangle: dict(corners=(u,v), tip=t) -> vertices (u, t, v)
    def add_tri(u, t, v):
        triangles.append((u, t, v))
        return len(triangles) - 1

    # ---- cell 0: ring of 6 triangles ----
    corners = [new() for _ in range(6)]            # ring corners k0..k5
    tips = [new() for _ in range(6)]               # tips p0..p5
    ring = [add_tri(corners[i], tips[i], corners[(i + 1) % 6]) for i in range(6)]
    # initial shared pair for the next cell: (S1, S2) = (t0, t5), common corner k0
    S1, S2 = ring[0], ring[5]

    def tip(tidx):
        return triangles[tidx][1]

    for _ in range(1, n_cells):
        # new cell: ring [S2, N1, N2, N3, N4, S1]
        # new corners between N1..N4; tip(S2) and tip(S1) become ring corners
        c_s2n1 = tip(S2)
        c_n4s1 = tip(S1)
        c12, c23, c34 = new(), new(), new()
        t1, t2, t3, t4 = new(), new(), new(), new()
        N1 = add_tri(c_s2n1, t1, c12)
        N2 = add_tri(c12, t2, c23)
        N3 = add_tri(c23, t3, c34)
        N4 = add_tri(c34, t4, c_n4s1)
        # shared pair for the next cell: the "opposite" adjacent pair (N2, N3),
        # whose common corner is c23  ->  the chain advances in a straight line
        S1, S2 = N3, N2

    edges = []
    for (a, b, c) in [(t[0], t[1], t[2]) for t in triangles]:
        edges += [[a, b], [b, c], [a, c]]
    # normalize (i<j) and verify uniqueness
    edges = [sorted(e) for e in edges]
    assert len({tuple(e) for e in edges}) == len(edges), "duplicate edges"

    # NOTE on the labeling: the labeling and the CREATION ORDER of the edges is kept
    # (triangle by triangle along the chain). Relabeling with reverse Cuthill-McKee
    # to reduce bandwidth (span 12 -> 4) was tried: it fixed the 47q/reps=1 case but
    # made 19q WORSE (truncation error 0.55 -> 6.3) and 26q too — Aer-MPS's internal
    # truncation depends on the labeling non-monotonically. The creation-order
    # labeling is the best found empirically.
    return nxt, edges, triangles


def dimer_cover(n, edges, seed=0):
    """
    Dimer cover via maximum-cardinality matching (networkx). Returns
    (dimers, spinons). NOTE: biasing the matching toward label-local pairs
    (weight 1000-|i-j|) to favor the MPS was tried; it produced covers that
    INTERACT WORSE with Aer's internal truncation cascade (26q went from an exact
    baseline to an error of 13). The unweighted matching, which is the empirically
    validated one, is kept.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([tuple(e) for e in edges])
    m = nx.max_weight_matching(G, maxcardinality=True)
    dimers = sorted(tuple(sorted(p)) for p in m)
    covered = {s for p in dimers for s in p}
    spinons = sorted(set(range(n)) - covered)
    return dimers, spinons


# --------------------------------------------------------------------------
# 2. Energy evaluator on MPS (symmetric ansatz)
# --------------------------------------------------------------------------
class MPSEnergyEvaluator:
    """
    SYMMETRIC HVA circuit (dimers + `reps` layers with one θ per edge class),
    transpiled once and evaluated with EstimatorV2 on
    AerSimulator(method='matrix_product_state').

    `max_bond_dimension` (χ) truncates the MPS. Our states are low-entanglement
    (product of singlets + a few local layers): χ=64 is generous, and the REAL
    memory stays bounded at ~ n·χ²·16 bytes (MBs). NOTE: Aer's memory validator
    estimates the WORST case ignoring χ (it requests ~10^8 MB at 47 qubits and
    aborts); that is why `max_memory_mb` is raised — it is only the validator's
    threshold, not reserved memory. Verified: one 47-qubit evaluation uses
    ~0.18 GB of peak RAM.
    """

    def __init__(self, n, edges, dimers, reps, max_bond_dimension=64):
        self.n, self.reps = n, reps
        self.edges = [tuple(e) for e in edges]
        self.colors, self.ncol = edge_color_classes(self.edges)
        self.n_params = reps * self.ncol
        self.H = heisenberg_hamiltonian(n, edges)

        qc = QuantumCircuit(n)
        for a, b in dimers:
            qc.h(a); qc.x(b); qc.cx(a, b); qc.z(a)
        self.P = ParameterVector("θ", self.n_params)
        for r in range(reps):
            for k, (i, j) in enumerate(self.edges):
                qc.append(heisenberg_gate(self.P[r * self.ncol + self.colors[k]]), [i, j])

        opts = {"method": "matrix_product_state",
                "matrix_product_state_max_bond_dimension": int(max_bond_dimension),
                "max_memory_mb": 10**12}  # validator threshold (see docstring); its
                                          # estimate grows with circuit depth
        backend = AerSimulator(**opts)
        self.qc_t = transpile(qc, backend, optimization_level=1)
        self.est = EstimatorV2(options={"backend_options": opts})
        self.nfev = 0

    def energy(self, x):
        self.nfev += 1
        res = self.est.run([(self.qc_t, self.H, np.asarray(x, dtype=float))]).result()
        return float(res[0].data.evs)


# --------------------------------------------------------------------------
# 3. Light exact reference (statevector without sparse matrices, n <= ~20)
# --------------------------------------------------------------------------
def light_exact_energy(n, edges, dimers, x, reps):
    """
    EXACT energy of the symmetric HVA via a "light" statevector: applies the gates
    with tensor contraction and sums bond energies term by term. It does not build
    the sparse Hamiltonian (which at 19 qubits weighs hundreds of MB): memory
    ~ 8·2^n bytes. Only for validating the MPS pipeline at n <= ~20.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from kagome_hva import heis_matrix
    E = [tuple(e) for e in edges]
    colors, ncol = edge_color_classes(E)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    BOND = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)

    def ap(psi, G, a, b):
        axa, axb = n - 1 - a, n - 1 - b
        psi = np.moveaxis(psi, [axa, axb], [0, 1])
        sh = psi.shape
        psi = (G @ psi.reshape(4, -1)).reshape(sh)
        return np.moveaxis(psi, [0, 1], [axa, axb])

    qc = QuantumCircuit(n)
    for a, b in dimers:
        qc.h(a); qc.x(b); qc.cx(a, b); qc.z(a)
    psi = Statevector(qc).data.reshape((2,) * n)
    for r in range(reps):
        for k, (i, j) in enumerate(E):
            psi = ap(psi, heis_matrix(x[r * ncol + colors[k]]), i, j)
    sv = psi.reshape(-1)
    return sum(float(np.real(np.vdot(sv, ap(psi.copy(), BOND, i, j).reshape(-1))))
               for (i, j) in E)


# --------------------------------------------------------------------------
# 4. Progressive (symmetric) sweep on MPS
# --------------------------------------------------------------------------
def run_mps_sweep_symmetric(n, edges, dimers, max_reps=3, n_random=1, maxiter=15,
                            seed=0, max_bond_dimension=64, chi_check=128,
                            method="L-BFGS-B", verbose=True):
    """
    Layer sweep of the symmetric HVA evaluated on MPS, with layer-by-layer warm start.

    Optimizer (chosen after diagnosis): 'L-BFGS-B' with FINITE-DIFFERENCE gradient
    (jac='2-point'; MPS energies are deterministic and smooth, so FD is reliable).
    COBYLA was ruled out: on 26 sites it did not escape the dimer plateau
    (gap 3.00 -> 2.96 over 3 layers), while FD-L-BFGS went -26.5 -> -28.0 in 10
    iterations on the generated 19-site lattice. `method='COBYLA'` remains available.

    Truncation protocol (honest): OPTIMIZE at χ=`max_bond_dimension` (fast;
    measured error ~0.3-0.5 on random states) and RE-EVALUATE each layer's optimum
    at χ=`chi_check` (measured error ~0.01): `energy_hi` is the quotable number, and
    |energy_hi - energy| estimates the truncation error.

    References without ED: lower bound -3T, dimers -3M, energy per site E/(4N).
    Returns a list of dicts (reps, n_params, energy, energy_hi, gap_lb, per_site, x,
    nfev, secs). gap_lb and per_site are computed with energy_hi when chi_check is on.
    """
    import time as _time
    rng = np.random.default_rng(seed)
    T = len(edges) // 3                      # in this family, every edge ∈ 1 triangle
    E_lb = -3.0 * T
    E_dimer = -3.0 * len(dimers)
    if verbose:
        print(f"[{n} sites | {len(edges)} bonds | {T} triangles | "
              f"{len(dimers)} dimers | defects T-M = {T - len(dimers)}]")
        print(f"references: lower bound -3T = {E_lb:.0f} | "
              f"static dimers -3M = {E_dimer:.0f} | "
              f"thermodynamic ~ {-0.4386*4*n:.1f} (indicative only, open boundary)")

    results = [dict(reps=0, n_params=0, energy=E_dimer, energy_hi=E_dimer,
                    gap_lb=E_dimer - E_lb, per_site=E_dimer / (4 * n),
                    x=np.array([]), nfev=0, secs=0.0)]
    if verbose:
        r = results[0]
        print(f"reps=0  static dimers       E={r['energy']:9.3f}  "
              f"gap_LB={r['gap_lb']:6.2f}  E/site={r['per_site']:.4f}")

    prev = np.array([])
    for reps in range(1, max_reps + 1):
        ev = MPSEnergyEvaluator(n, edges, dimers, reps,
                                max_bond_dimension=max_bond_dimension)
        inits = []
        if prev.size:
            inits.append(np.concatenate([prev, rng.uniform(-0.05, 0.05, ev.ncol)]))
        for _ in range(n_random):
            inits.append(rng.uniform(-0.5, 0.5, ev.n_params))

        t0 = _time.time()
        best, bestx = np.inf, None
        maxfun = maxiter * (ev.n_params + 3)      # hard cap on evaluations: keeps
        for x0 in inits:                          # noisy line searches from running away
            if method == "L-BFGS-B":
                res = minimize(ev.energy, x0, method="L-BFGS-B", jac="2-point",
                               options={"maxiter": maxiter, "maxfun": maxfun,
                                        "finite_diff_rel_step": 1e-6})
            else:
                res = minimize(ev.energy, x0, method="COBYLA",
                               options={"maxiter": maxiter, "rhobeg": 0.4})
            if res.fun < best:
                best, bestx = res.fun, res.x

        # --- monotonicity guarantee: a new layer NEVER reports worse than the previous ---
        # (the previous solution with the new layer set to identity reproduces the same
        # state; if the optimization does not beat it, it is kept — the sweep stays
        # monotone)
        x_keep = np.concatenate([results[-1]["x"], np.zeros(ev.ncol)]) \
            if results[-1]["x"].size else np.zeros(ev.n_params)
        e_keep = results[-1]["energy"]
        if best >= e_keep:
            best, bestx = e_keep, x_keep
            if verbose:
                print(f"reps={reps}: no improvement over previous layer -> kept "
                      f"(E={e_keep:.3f})")
        prev = bestx
        secs = _time.time() - t0

        e_hi = float(best)
        if chi_check and chi_check > max_bond_dimension:
            ev_hi = MPSEnergyEvaluator(n, edges, dimers, reps,
                                       max_bond_dimension=chi_check)
            e_hi = ev_hi.energy(bestx)

        rec = dict(reps=reps, n_params=ev.n_params, energy=float(best),
                   energy_hi=e_hi, gap_lb=e_hi - E_lb, per_site=e_hi / (4 * n),
                   x=bestx, nfev=ev.nfev, secs=secs)
        results.append(rec)
        if verbose:
            chi_msg = (f"  E(χ={chi_check})={e_hi:9.3f} [trunc {abs(e_hi-best):.3f}]"
                       if chi_check and chi_check > max_bond_dimension else "")
            print(f"reps={reps}  params={ev.n_params:3d}          E={best:9.3f}{chi_msg}  "
                  f"gap_LB={rec['gap_lb']:6.2f}  E/site={rec['per_site']:.4f}  "
                  f"({ev.nfev} evals, {secs:.0f}s)")
    return results
