"""
kagome_common.py
-----------------
Utilidades compartidas para reproducir los resultados de:

Ahsan, M. "Utility-scale Experimental Quantum Computation with Hardware
Efficient Ansatze and Calibrated Hamiltonian", ACM Trans. Quantum Comput. (2026)

Convención del paper (Sección 3.1, ecuación 1):
    H = J * sum_{(a,b) in bonds} (X_a X_b + Y_a Y_b + Z_a Z_b)

Con esta normalización (factor "4x" mencionado en el paper), la energía de un
singlete en un bond individual es -3 y la de un triplete es +1 (en vez de
-3/4 y +1/4 de la convención estándar S_i . S_j).

Requiere: qiskit >= 2.0, scipy, numpy
    pip install qiskit scipy numpy
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import RealAmplitudes
from scipy.optimize import minimize


def build_hamiltonian(n_qubits, bonds):
    """
    Construye H = sum_bonds J*(XiXj + YiYj + ZiZj) como SparsePauliOp.

    bonds : lista de tuplas (i, j, J) -- sitios i,j (0-indexed) y su
            acoplamiento J (J=1 para bonds normales, J' para bonds calibrados).
    """
    terms, coeffs = [], []
    for (i, j, J) in bonds:
        for P in ['X', 'Y', 'Z']:
            label = ['I'] * n_qubits
            label[n_qubits - 1 - i] = P   # Qiskit usa orden little-endian
            label[n_qubits - 1 - j] = P
            terms.append(''.join(label))
            coeffs.append(J)
    return SparsePauliOp(terms, coeffs).simplify()


def exact_gs_energy(H, sparse_threshold=12):
    """
    Diagonalización exacta del Hamiltoniano.
    - Para n_qubits pequeño (<=12): diagonalización densa completa (numpy).
    - Para n_qubits mayor: solo el autovalor más bajo vía Lanczos disperso
      (scipy.sparse.linalg.eigsh), para evitar quedarse sin memoria.
    """
    n = H.num_qubits
    if n <= sparse_threshold:
        mat = H.to_matrix()
        evals = np.linalg.eigvalsh(mat)
        return evals[0]
    else:
        from scipy.sparse.linalg import eigsh
        mat = H.to_matrix(sparse=True)
        evals = eigsh(mat, k=1, which='SA', maxiter=5000, tol=1e-10,
                       return_eigenvectors=False)
        return evals[0]


def real_amplitudes_ansatz(n_qubits, reps=1):
    """
    El ansatz hardware-efficient del paper (Fig. 1b, Fig. 9a):
    capa de Ry(theta) -> escalera lineal de CNOT -> capa final de Ry(theta).
    Corresponde exactamente a qiskit.circuit.library.RealAmplitudes(reps=1,
    entanglement='linear').
    """
    return RealAmplitudes(n_qubits, reps=reps, entanglement='linear')


def vqe_energy(H, n_qubits, reps=1, n_restarts=8, seed=0, maxiter=400,
               method='L-BFGS-B'):
    """
    VQE clásico (statevector, sin ruido) sobre el ansatz de arriba.

    method: 'L-BFGS-B' (rápido para sistemas chicos, usa diferencias finitas)
            'COBYLA'   (más barato por iteración; recomendado para >12 qubits)

    Devuelve (mejor_energia, mejores_parametros, ansatz).
    """
    ansatz = real_amplitudes_ansatz(n_qubits, reps=reps)
    nparams = ansatz.num_parameters
    rng = np.random.default_rng(seed)

    def cost(theta):
        bound = ansatz.assign_parameters(theta)
        sv = Statevector.from_instruction(bound)
        return np.real(sv.expectation_value(H))

    best, best_val = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform(0, 2 * np.pi, size=nparams)
        if method == 'COBYLA':
            res = minimize(cost, x0, method='COBYLA',
                            options={'maxiter': maxiter, 'tol': 1e-8})
        else:
            res = minimize(cost, x0, method='L-BFGS-B',
                            options={'maxiter': maxiter, 'ftol': 1e-12,
                                     'gtol': 1e-10})
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    return best_val, best, ansatz


def bond_energy_map(bonds, n_qubits, theta, ansatz):
    """
    Calcula <psi| (XiXj+YiYj+ZiZj) |psi> bond por bond, para el estado
    preparado por `ansatz` con parámetros `theta`.  Reproduce los mapas de
    energía de bond de la Fig. 1(c) del paper.

    bonds: lista de (i, j, J) -- J se usa solo como referencia/etiqueta,
           el bond individual siempre se evalúa con coeficiente 1
           salvo que se pase explícitamente J.
    """
    bound = ansatz.assign_parameters(theta)
    sv = Statevector.from_instruction(bound)
    out = {}
    for (i, j, J) in bonds:
        Hij = build_hamiltonian(n_qubits, [(i, j, J)])
        out[(i, j)] = np.real(sv.expectation_value(Hij))
    return out
