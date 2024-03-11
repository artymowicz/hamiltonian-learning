import numpy as np
import scipy
import mosek
from mosek.fusion import Model, Domain, Matrix, Expr, ObjectiveSense
import sys
import utils

def dag(m):
	return np.conjugate(m.T)

'''
INTERFACE FOR learnHamiltonianFromThermalState

INPUT:

	int r                          |  number of perturbing operators
	int s                          |  number of variational hamiltonian terms
	ndarray[float] h_terms_exp     |  length-s vector of floats which contains the expectations of the hamiltonian terms, ie. omega(h_alpha) for alpha = 1,...,s
	ndarray[complex] J             |  r x r complex matrix satisfying b_i^* = sum_j J_ij b_j (where b_i are the perturbing operators)
	ndarray[complex] C             |  r x r complex matrix given by C_ij = omega(b_i^* b_j)

	ndarray[int] F_indices         |  sparse representation (in COO format) of the tensor with 
	ndarray[complex] F_values      |   dimensions (r,r,s) given by F_ijk = omega(b_i^* [h_k, b_j])

	float epsilon_W                |  singular value threshold for computing the approximate kernel of the matrix W
	int printing_level             |  how verbose the program is to be. Either 0,1, or 2.

OUTPUT:

	ndarray[float] hamiltonian_coefficients
	float T
	float mu
	int q

'''

def learnHamiltonianFromThermalState(r: int, s: int, h_terms_exp: np.ndarray, J: np.ndarray, C: np.ndarray, F_indices: np.ndarray , F_values: np.ndarray,  
		epsilon_W: float, printing_level: int) -> tuple[np.ndarray, float, float, int]:

	F = utils.SparseTensor((r,r,s), F_indices, F_values)
	F_vectorized = F.vectorize(axes = [0,1]).toScipySparse()

	###---------- step 1: diagonalizing the covariance matrix C ----------###

	C_eigvals, C_eigvecs = scipy.linalg.eigh(C)

	small = 1e-9 
	very_small = 1e-15

	if C_eigvals[0] < very_small:
		raise ValueError('covariance matrix is singular. The given expectation values do not correspond to a Gibbs state')
	elif C_eigvals[0] < small:
		utils.tprint('WARNING: covariance matrix is near singular')

	###---------- step 2: computing quasi-symmetries ----------###

	### building W matrix
	if printing_level > 2:
		utils.tprint('building W matrix')
	F_dag = F.conjugate().transpose([1,0,2])
	Z = F - F_dag
	Z_vec = Z.vectorizeLowerTriangular([0,1])
	Z_vec_scipy = Z_vec.toScipySparse()
	W = (Z_vec_scipy.T.conj()@Z_vec_scipy)
	W = W.toarray()

	### computing the matrix S, whose columns form an orthonormal basis of the (approximate) null space of W
	W_eigvals, W_eigvecs = scipy.linalg.eigh(W)
	q = 0
	while q < s:
		if W_eigvals[q] > epsilon_W:
			break
		else:
			q += 1
	if printing_level > 1:
		utils.tprint('lowest 10 eigenvalues of W:')
		for i in range(10):
			utils.tprint(f'  {W_eigvals[i]:.4e}')
	if q < 1 :
		raise ValueError(f'all W eigenvalues are above the cutoff {epsilon_W}')
	if printing_level > 1:
		utils.tprint(f'W eigenvalue threshold = {epsilon_W}')
	if printing_level > 1:
		utils.tprint(f'approximate kernel of W has dimension {q}')
	S = W_eigvecs[:,:q] 

	### S is real so we cast it to float
	assert np.array_equal(np.imag(S), np.zeros(S.shape))
	S = np.asarray(np.real(S), dtype = 'float64')

	###---------- step 3: convex optimization ----------###

	E = C_eigvecs@np.diag(np.reciprocal(np.sqrt(C_eigvals)))
	Delta = dag(E)@J@C.T@dag(J)@E
	
	
	Delta_eigvals = scipy.linalg.eigvalsh(Delta)

	if Delta_eigvals[0] < very_small:
		raise valueError('modular operator is singular. The given expectation values do not correspond to a Gibbs state')
	elif Delta_eigvals[0] < small:
		utils.tprint('WARNING: Modular operator is near singular')
	

	logDelta = scipy.linalg.logm(Delta)

	if printing_level > 0:
		utils.tprint('learning Hamiltonian')

	with Model() as M:
		M.setLogHandler(sys.stdout)
		if printing_level > 2:
			M.setSolverParam("log",10000)
		else:
			M.setSolverParam("log",0)

		L = M.variable("L", Domain.inPSDCone(2*r))
		nu = M.variable("nu")

		M.objective(ObjectiveSense.Maximize, Expr.mul(-1,nu))

		### precomputing the tensor contraction of F with S. In the terminology of the paper, this computes \tilde{\boldsymbol{H}}_{alpha} as a rank-3 tensor
		F_real = F.complexToReal([0,1])
		F_real_vectorized = F_real.vectorize([0,1]).toScipySparse()
		FS = (F_real_vectorized@S).reshape((2*r,2*r,q), order = 'F')

		### enforcing the first constraint (whose dual variable is the Hamiltonian)
		E_real = utils.complexToReal(E)
		omega_h_tilde = h_terms_exp@S
		H_tilde_alphas = [np.asarray(E_real.T@FS[:,:,i]@E_real, dtype = 'float64') for i in range(q)]

		expr = Expr.vstack([Expr.add(Expr.dot(H_tilde_alphas[i],L), Expr.mul(nu, omega_h_tilde[i])) for i in range(q)])
		M.constraint('tilde_h_alpha', expr , Domain.equalsTo(0))

		### enforcing the second constraint (whose dual variable is the temperature)
		M.constraint("logDelta_dot", Expr.dot(utils.complexToReal(logDelta), L), Domain.lessThan(0))

		### enforcing the third constraint (whose dual variable is mu)
		M.constraint("trace_one", Expr.mul(1,Expr.dot(Matrix.eye(2*r),L)), Domain.equalsTo(1))

		if printing_level > 2:
			utils.tprint('running MOSEK:')
			print()

		M.solve()

		if printing_level > 2:
			print()

		### recovering the optimal values of the primal program as the dual variables of the dual program
		hamiltonian_coefficients = S@M.getConstraint("tilde_h_alpha").dual()
		T = M.getConstraint("logDelta_dot").dual()[0]
		mu = M.getConstraint("trace_one").dual()[0]

	return hamiltonian_coefficients, T, mu, q
