import numpy as np
import scipy
import cvxpy as cp
import argparse
import utils
import matplotlib.pyplot as plt
from matplotlib import colormaps
import mosek
#from mosek.fusion import Mode, Domain, Matrix, Expr, ObjectiveSense
from mosek.fusion import *

import sys

def plotSpec(*matrices, hermitean = True, names = None, title = None, xscale = 'linear', yscale = 'linear'):
	cmap = colormaps['viridis']

	if names is not None:
		assert len(matrices) == len(names)
	else:
		names = [None]*len(matrices)

	for i in range(len(matrices)):
		if hermitean:
			eigs = scipy.linalg.eigvalsh(matrices[i])
			plt.scatter(np.arange(len(eigs)), eigs, s=2, label = names[i])
		else:
			eigs =  scipy.linalg.eigvals(matrices[i])
			plt.scatter(np.arange(len(eigs)), np.real(eigs), s=2, label = names[i] + 'real', c =cmap(i/len(matrices)) )
			plt.scatter(np.arange(len(eigs)), np.imag(eigs), marker = "^", s=2, label = names[i] + 'imaginary',c =cmap(i/len(matrices)))

	plt.title(title)
	plt.yscale(yscale)
	plt.xscale(xscale)
	plt.legend()
	plt.show()

def learnHamiltonianFromGroundstate(n, onebody_operators, hamiltonian_terms, expectations_evaluator, params_dict, metrics):
	# I don't think this works
	if type(expectations_evaluator) == dict:
		expectations_evaluator = lambda x: expectations_evaluator[x]

	if params_dict['printing']:
		utils.tprint('building multiplication tensor')
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)

	if params_dict['printing']:
		utils.tprint('building threebody operators')
	threebody_operators = utils.buildThreeBodyTermsFast(onebody_operators, hamiltonian_terms, params_dict['printing'])

	if params_dict['printing']:
		utils.tprint('building triple product tensor')
	triple_product_tensor = utils.buildTripleProductTensorFast(onebody_operators, hamiltonian_terms, threebody_operators, params_dict['printing'])

	try:
		hamiltonian_coefficients_expectations = [expectations_evaluator(x) for x in hamiltonian_terms]
		twobody_expectations =[expectations_evaluator(x) for x in twobody_operators]
		threebody_expectations = [expectations_evaluator(x) for x in threebody_operators]
	except KeyError as inst:
		print(f'learnHamiltonianFromGroundstate was not given enough expectations. Missing operator: {str(inst)}')
		raise

	if params_dict['printing']:
		utils.tprint('building F tensor')
	triple_product_tensor = triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	F = triple_product_tensor.contractRight(threebody_expectations)#F_ijk = <a_i[a_k,a_j]> (note the order of indices)
	F_vectorized = F.vectorize(axes = [0,1]).toScipySparse()

	if params_dict['printing']:
		utils.tprint('building covariance matrix')
	l = len(onebody_operators)
	h = len(hamiltonian_terms)
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()#C_ij = <a_ia_j>

	if params_dict['printing']:
		utils.tprint('learning Hamiltonian')
	X = cp.Variable(h)
	constraints = [X[0]==1, X@hamiltonian_coefficients_expectations == 0, cp.reshape(F_vectorized@X, (l,l))>>0]
	#g = 0#np.random.normal(size = (l,))
	if params_dict['objective'] == 'l2':
		objective = cp.square(cp.norm(X,2))
	elif params_dict['objective'] == 'l1':
		objective = cp.norm(X,1)
	elif params_dict['objective'] == 'rand_linear':
		objective = X@np.random.normal(size = h)
	else:
		raise ValueError(f"params_dict['objective'] unrecognized: {params_dict['objective']}")
	
	prob = cp.Problem(cp.Minimize(objective), constraints)
	prob.solve(solver = 'MOSEK', verbose = params_dict['printing'])#, save_file = 'dump.ptf')

	coeffs = X.value

	solver_stats = {}
	#solver_stats['compilation_time'] = prob.compilation_time
	solver_stats['status'] = prob.status
	solver_stats['extra_stats'] = prob.solver_stats.extra_stats
	solver_stats['num_iters'] = prob.solver_stats.num_iters
	solver_stats['solve_time'] = prob.solver_stats.solve_time
	solver_stats['solver_name'] = prob.solver_stats.solver_name
	metrics['solver_stats'] = solver_stats

	return X.value, C, F

### deprecated
def discardZeroEgivals(m, threshold = 0):
	eigvals, eigvecs = scipy.linalg.eigh(m)
	print(eigvals)
	for i in range(l):
		if eigvals[i] > threshold:
			cutoff = i
			break
	eigvecs_restricted = eigvecs[:,cutoff:]
	return np.conjugate(eigvecs.T)@diag(eigvals[cutoff:])@eigvecs

def streamprinter(text):
	sys.stdout.write(text)
	sys.stdout.flush()

def cvxSolve(logDelta,F_vectorized,W, hamiltonian_terms_expectations, params_dict, extras, metrics):
	#F_lt_vec = F.copy().complexToReal([0,1])
	#F_lt_vec.vectorizeLowerTriangular([0,1], strict = False, scale_off_diagonals = np.sqrt(2))
	#print(F_lt_vec.shape)
	#F_lt_vec = F_lt_vec.toScipySparse()

	
	#a = utils.vectorizeLowerTriangular(utils.complexToReal(logDelta), scale_off_diagonals = np.sqrt(2), strict = False).reshape((mat_dim,1))
	#b = utils.vectorizeLowerTriangular(np.eye(2*r), strict = False).reshape((mat_dim,1))
	#print((FS.shape,a.shape,b.shape))
	#FS_tilde = np.hstack((FS,a,b))

	#bottom_row = np.zeros(q+2)
	#bottom_row[q]=1.
	#FS_tilde = np.vstack((FS_tilde, bottom_row))

	

	return S@X.value, T.value, mu.value

### minimizes mu subject to T @ logDelta + F@X >> -mu and T >= 0
def mosekSolve(logDelta, F, W):
	r, _ = logDelta.shape
	W_eigvals, W_eigvecs = scipy.linalg.eigh(W)
	threshold = 1e-10
	q = list((W_eigvals > threshold)).index(True)
	if q == 0:
		raise ValueError('all W eigenvalues are above the cutoff')
	print(f'W cutoff = {q}')
	S = W_eigvecs[:,:q]

	#TODO: check validity of this
	assert np.linalg.norm(np.imag(S)) < 1e-10
	S = np.real(S)

	F_lt_vec = F.complexToReal([0,1]).vectorizeLowerTriangular([0,1], strict = False, scale_off_diagonals = np.sqrt(2))
	print(F_lt_vec.shape)
	F_lt_vec = F_lt_vec.toScipySparse()
	FS = F_lt_vec@S
	mat_dim = (2*r*(2*r+1))//2
	a = utils.vectorizeLowerTriangular(utils.complexToReal(logDelta), scale_off_diagonals = np.sqrt(2), strict = False).reshape((mat_dim,1))
	b = utils.vectorizeLowerTriangular(np.eye(2*r), strict = False).reshape((mat_dim,1))
	print((FS.shape,a.shape,b.shape))
	FS_tilde = np.hstack((FS,a,b))

	#bottom_row = np.zeros(q+2)
	#bottom_row[q]=1.
	#FS_tilde = np.vstack((FS_tilde, bottom_row))

	print(FS_tilde.shape)
	c = np.zeros(q+2)
	c[q+1] = 1

	FS_tilde_sparse = utils.SparseTensor(FS_tilde)
	print(FS_tilde_sparse)
	print(FS_tilde.shape)

	with mosek.Task() as task:
		task.set_Stream(mosek.streamtype.log, streamprinter)

		# Below is the sparse triplet representation of the F matrix.
		afeidx = FS_tilde_sparse.indices[:,0]
		varidx = FS_tilde_sparse.indices[:,1]
		f_val  = FS_tilde_sparse.values

		#barcj = [0, 0]
		#barck = [0, 1]
		#barcl = [0, 1]
		#barcv = [1, 1]

		#barfi = [0,0]
		#barfj = [0,0]
		#barfk = [0,1]
		#barfl = [0,0]
		#barfv = [0,1]

		numvar = q+2
		numafe = mat_dim
		#BARVARDIM = [2]

		# Append 'numvar' variables.
		# The variables will initially be fixed at zero (x=0).
		task.appendvars(numvar)

		# Append 'numafe' empty affine expressions.
		task.appendafes(numafe)

		# Append matrix variables of sizes in 'BARVARDIM'.
		# The variables will initially be fixed at zero.
		#task.appendbarvars(BARVARDIM)

		# Set the linear terms in the objective.
		task.putcj(q+1, 1.0)
		#task.putcfix(1.0)
		#task.putbarcblocktriplet(barcj, barck, barcl, barcv)

		for j in range(q):
			# Set the bounds on variable j
			# blx[j] <= x_j <= bux[j]
			task.putvarbound(j, mosek.boundkey.fr, 0, 0) #with mosek.boundkey.fr the upper and lower bounds arent read 

		task.putvarbound(q, mosek.boundkey.lo, 0, 0) #T > 0 #let's see if this works -- the second zero should be ignored
		task.putvarbound(q+1, mosek.boundkey.fr, 0, 0) #mu is free

		# Set up the F matrix of the problem
		task.putafefentrylist(afeidx, varidx, f_val)
		# Set up the g vector of the problem
		#task.putafegslice(0, numafe, g)
		#task.putafebarfblocktriplet(barfi, barfj, barfk, barfl, barfv)

		# Append R+ domain and the corresponding ACC
		#task.appendacc(task.appendrplusdomain(1), [0], None) 
		# Append SVEC_PSD domain and the corresponding ACC
		task.appendacc(task.appendsvecpsdconedomain(mat_dim), list(range(mat_dim)), None)

		# Input the objective sense (minimize/maximize)
		task.putobjsense(mosek.objsense.minimize)

		# Solve the problem and print summary
		task.optimize()
		task.solutionsummary(mosek.streamtype.msg)

		# Get status information about the solution
		prosta = task.getprosta(mosek.soltype.itr)
		solsta = task.getsolsta(mosek.soltype.itr)

		if (solsta == mosek.solsta.optimal):
			xx = task.getxx(mosek.soltype.itr)
			print("Optimal solution:\nx=%s\n" % (xx))
		elif (solsta == mosek.solsta.dual_infeas_cer or
			  solsta == mosek.solsta.prim_infeas_cer):
			print("Primal or dual infeasibility certificate found.\n")
		elif solsta == mosek.solsta.unknown:
			print("Unknown solution status")
		else:
			print("Other solution status")

	return X_value, mu_value, T_value, solver_stats

### minimizes mu subject to T @ logDelta + F@X >> -mu and T >= 0
def mosekSolveNew(logDelta, F, W, threshold):
	

	#bottom_row = np.zeros(q+2)
	#bottom_row[q]=1.
	#FS_tilde = np.vstack((FS_tilde, bottom_row))

	print(FS_tilde.shape)
	c = np.zeros(q+2)
	c[q+1] = 1

	FS_tilde_sparse = utils.SparseTensor(FS_tilde)
	print(FS_tilde_sparse)
	print(FS_tilde.shape)

	with mosek.Task() as task:
		task.set_Stream(mosek.streamtype.log, streamprinter)

		# Below is the sparse triplet representation of the F matrix.
		afeidx = FS_tilde_sparse.indices[:,0]
		varidx = FS_tilde_sparse.indices[:,1]
		f_val  = FS_tilde_sparse.values

		#barcj = [0, 0]
		#barck = [0, 1]
		#barcl = [0, 1]
		#barcv = [1, 1]

		#barfi = [0,0]
		#barfj = [0,0]
		#barfk = [0,1]
		#barfl = [0,0]
		#barfv = [0,1]

		numvar = q+2
		numafe = mat_dim
		#BARVARDIM = [2]

		# Append 'numvar' variables.
		# The variables will initially be fixed at zero (x=0).
		task.appendvars(numvar)

		# Append 'numafe' empty affine expressions.
		task.appendafes(numafe)

		# Append matrix variables of sizes in 'BARVARDIM'.
		# The variables will initially be fixed at zero.
		#task.appendbarvars(BARVARDIM)

		# Set the linear terms in the objective.
		task.putcj(q+1, 1.0)
		#task.putcfix(1.0)
		#task.putbarcblocktriplet(barcj, barck, barcl, barcv)

		for j in range(q):
			# Set the bounds on variable j
			# blx[j] <= x_j <= bux[j]
			task.putvarbound(j, mosek.boundkey.fr, 0, 0) #with mosek.boundkey.fr the upper and lower bounds arent read 

		task.putvarbound(q, mosek.boundkey.lo, 0, 0) #T > 0 #let's see if this works -- the second zero should be ignored
		task.putvarbound(q+1, mosek.boundkey.fr, 0, 0) #mu is free

		# Set up the F matrix of the problem
		task.putafefentrylist(afeidx, varidx, f_val)
		# Set up the g vector of the problem
		#task.putafegslice(0, numafe, g)
		#task.putafebarfblocktriplet(barfi, barfj, barfk, barfl, barfv)

		# Append R+ domain and the corresponding ACC
		#task.appendacc(task.appendrplusdomain(1), [0], None) 
		# Append SVEC_PSD domain and the corresponding ACC
		task.appendacc(task.appendsvecpsdconedomain(mat_dim), list(range(mat_dim)), None)

		# Input the objective sense (minimize/maximize)
		task.putobjsense(mosek.objsense.minimize)

		# Solve the problem and print summary
		task.optimize()
		task.solutionsummary(mosek.streamtype.msg)

		# Get status information about the solution
		prosta = task.getprosta(mosek.soltype.itr)
		solsta = task.getsolsta(mosek.soltype.itr)

		if (solsta == mosek.solsta.optimal):
			xx = task.getxx(mosek.soltype.itr)
			print("Optimal solution:\nx=%s\n" % (xx))
		elif (solsta == mosek.solsta.dual_infeas_cer or
			  solsta == mosek.solsta.prim_infeas_cer):
			print("Primal or dual infeasibility certificate found.\n")
		elif solsta == mosek.solsta.unknown:
			print("Unknown solution status")
		else:
			print("Other solution status")

	return X_value, mu_value, T_value, solver_stats

def parseObjective(X,T,mu, S, params_dict):
	if params_dict['objective'] == 'l2':
		objective = cp.square(cp.norm(S@X,2))
	elif params_dict['objective'] == 'l1':
		objective = cp.norm(S@X,1)
	elif params_dict['objective'] == 'rand_linear':
		objective = (np.random.normal(size = h)@S)@X
	elif params_dict['objective'] == 'T':
		if params_dict['T_constraint'] == 'T=1':
			print('setting T_constraint to "T>0" because objective_constraint is "T"')
			params_dict['T_constraint'] == 'T>0'
		objective = T
	elif params_dict['objective'] == 'minus_T':
		if params_dict['T_constraint'] == 'T=1':
			print('setting T_constraint to "T>0" because objective_constraint is "minus_T"')
			params_dict['T_constraint'] == 'T>0'
		objective = -T
	elif params_dict['objective'] == 'mu':
		objective = mu
	else:
		raise ValueError(f"objective {params_dict['objective']} not recognized. Valid inputs are 'l1', 'l2, 'rand_linear', 'T', or 'minus_T'")
	return objective

def parseConstraints(X,T,mu,S, hamiltonian_terms_expectations, params_dict):
	return 

'''
params_dict required arguments:

threshold: the threshold for computing the inverse of covariance matrix
mu: a small offset added to the positivity of the free energy
'''

def learnHamiltonianFromThermalState(n, onebody_operators, hamiltonian_terms, expectations_evaluator, params_dict, metrics, return_extras = False):
	### throughout, I assume that the identity is the first entry in hamiltonian_terms. It's used in the normalization
	assert hamiltonian_terms[0]=='I'*n
	hamiltonian_terms = hamiltonian_terms[1:] #TESTING
	#print(f'len of hamiltonian_terms: {len(hamiltonian_terms)}')
	print('hamiltonian_terms:')
	for x in hamiltonian_terms:
		print(f'{x} {expectations_evaluator(x)}')
	print(f'onebody_operators = {onebody_operators}')

	### building multiplication tensor
	if type(expectations_evaluator) == dict:
		expectations_evaluator = lambda x: expectations_evaluator[x]

	if params_dict['printing']:
		utils.tprint('building multiplication tensor')
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)

	if params_dict['printing']:
		utils.tprint('building threebody operators')
	threebody_operators = utils.buildThreeBodyTermsFast(onebody_operators, hamiltonian_terms, params_dict['printing'])

	if params_dict['printing']:
		utils.tprint('building triple product tensor')
	triple_product_tensor = utils.buildTripleProductTensorFast(onebody_operators, hamiltonian_terms, threebody_operators, params_dict['printing'])

	try:
		hamiltonian_terms_expectations = [expectations_evaluator(x) for x in hamiltonian_terms]
		twobody_expectations =[expectations_evaluator(x) for x in twobody_operators]
		threebody_expectations = [expectations_evaluator(x) for x in threebody_operators]
	except KeyError as inst:
		print(f'learnHamiltonianFromThermalstate was not given enough expectations. Missing operator: {str(inst)}')
		raise

	R = len(onebody_operators)
	h = len(hamiltonian_terms)

	### computing F tensor
	if params_dict['printing']:
		utils.tprint('building F tensor')

	triple_product_tensor = triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	F = triple_product_tensor.contractRight(threebody_expectations)#F_ijk = <a_i[h_k,a_j]> (note the order of indices)
	F_vectorized = F.vectorize(axes = [0,1]).toScipySparse()

	### computing covariance matrix C and modular operator Delta
	if params_dict['printing']:
		utils.tprint('computing covariance matrix (C) and modular operator (Delta)')
	
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()#C_ij = <a_ia_j>
	C_eigvals, C_eigvecs = scipy.linalg.eigh(C)

	# compute how many of the lowest eigenvalues of C to throw out
	C_eigval_cutoff = R
	for i in range(R):
		if C_eigvals[i] > params_dict['C_eigval_threshold']:
			C_eigval_cutoff = i
			break
	if C_eigval_cutoff == R:
		raise RuntimeError(f"no C eigenvalues were above threshold {params_dict['C_eigval_threshold']}")

	metrics['C_eigval_cutoff'] = C_eigval_cutoff
	if params_dict['printing']:
		utils.tprint(f'threw out the lowest {C_eigval_cutoff} eigenvalues of C')

	r = R - C_eigval_cutoff

	D = np.diag(np.reciprocal(np.sqrt(C_eigvals[C_eigval_cutoff:]))) #D = C^{-1/2}
	E = C_eigvecs[:,C_eigval_cutoff:]@D #E is D acting on the orthocomplement of the (approximate) kernel of C
	Delta = np.conjugate(E.T)@C.T@E
	
	Delta_eigs = scipy.linalg.eigvalsh(Delta)
	small = 1e-9 
	very_small = 1e-11
	set_T_zero_flag = False
	if Delta_eigs[0] < very_small:
		utils.tprint('Modular operator is singular. Setting T to zero and objective to l1 ')
		set_T_zero_flag = True
	elif Delta_eigs[0] < small:
		utils.tprint('WARNING: Modular operator is near singular')
	logDelta = scipy.linalg.logm(Delta)
	#TODO: just project onto Ker(Delta)^perp. In fact this should be automatic anyway
	#utils.plotSpec(logDelta, names = ['logDelta'], print_lowest = 10)

	### building W and S matrices
	utils.tprint('creating F_dag')
	F_dag = F.conjugate().transpose([1,0,2])
	utils.tprint('subtracting F from F_dag')
	Z = F - F_dag
	utils.tprint('vectorizing Z')
	Z_vec = Z.vectorizeLowerTriangular([0,1], order = False)
	utils.tprint('converting Z to scipy sparse')
	Z_vec_scipy = Z_vec.toScipySparse()
	utils.tprint('multiplying Z dag with Z')
	W = (Z_vec_scipy.T.conj()@Z_vec_scipy)
	utils.tprint('converting W to dense array')
	W = W.toarray()

	## computing S, whose columns form an orthonormal basis of the (approximate) null space of W
	W_eigvals, W_eigvecs = scipy.linalg.eigh(W)
	W_eigval_threshold = params_dict['W_eigval_threshold']
	q = list((W_eigvals > W_eigval_threshold)).index(True)
	if params_dict['printing']:
		utils.tprint('lowest 10 eigenvalues of W:')
		for i in range(10):
			utils.tprint(f' {W_eigvals[i]:.4e}')
	if q < 2 :
		raise ValueError(f'all W eigenvalues are above the cutoff {W_eigval_threshold} (except the trivial one coming from the identity)')
	utils.tprint(f'W cutoff = {q}')
	S = W_eigvecs[:,:q] 

	#TODO: check validity of this. It seems like W is always real and symmetric
	#imag(S) is exactly zero (no roundoff errors), even with noise in the expectations. So probably a good reason
	assert np.array_equal(np.imag(S), np.zeros(S.shape))
	S = np.real(S)

	###----------the convex optimization happens here----------###

	if params_dict['printing']:
		utils.tprint('learning Hamiltonian')

	X = cp.Variable(q)
	T = cp.Variable()
	mu = cp.Variable()

	#rootC = scipy.linalg.sqrtm(C)
	#g = lambda T, logDelta, E, F_vectorized, X : -T*rootC@scipy.linalg.logm(rootC@scipy.linalg.inv(C.T)@rootC)@rootC + cp.reshape(F_vectorized@X, (r,r))
	#constraints += [g(T, logDelta, E, F_vectorized, X) >> -params_dict['mu']]
	#constraints += [T*D_inv@logDelta@D_inv + np.conjugate(eigvecs[:,cutoff:].T)@cp.reshape(F_vectorized@X, (r,r))@eigvecs[:,cutoff:] >> 0 ]
	#g = 0#np.random.normal(size = (l,))
	#objective = cp.sum(cp.square(g-X))

	### building constraints
	if params_dict['T_constraint'] == "T>0":
		constraints = [T>=0]
		h,_ = S.shape
		#u = np.zeros(h)
		#u[0] = 1
		#v = S.T@u
		#constraints += [v@X == 1]
		w = S.T@hamiltonian_terms_expectations
		constraints += [w@X == -1]

	elif params_dict['T_constraint'] == 'T=1':
		constraints = [T==1, X[0] == 0]
		h,_ = S.shape
		u = np.zeros(h)
		u[0] = 1
		v = S.T@u
		constraints += [v@X == 0]
	else:
		raise ValueError(f"T_constraint {params_dict['T_constraint']} not recognized. Valid inputs are 'T=1' or 'T>0'.")

	if params_dict['objective'] != 'mu':
		constraints += [mu == params_dict['mu']]

	FS = F_vectorized@S
	print(f'tidle_h_alpha_2 = {S[:,2]}')

	##testing:
	tilde_H_alpha_2 = FS[:,2].reshape((r,r), order='F')
	print(f'tilde_H_alpha_2.shape = {tilde_H_alpha_2.shape}')
	for i in range(r):
		for j in range(r):
			if np.abs(tilde_H_alpha_2[i,j]) > 1e-10:
				print(f'tilde_H_alpha_2 entry at {i,j} = {tilde_H_alpha_2[i,j]}')
	print(f'omega_tilde = {np.array([expectations_evaluator(x) for x in hamiltonian_terms])@S}')

	if set_T_zero_flag:
		constraints += [T == 0, mu == 0]
		params_dict['objective'] = 'l1'
		constraints += [np.conjugate(E.T)@cp.reshape(FS@X, (R,R))@E + np.eye(r)*mu >> 0]
	else:
		constraints += [T*logDelta + np.conjugate(E.T)@cp.reshape(FS@X, (R,R))@E + np.eye(r)*mu >> 0]

	### building objective
	objective = parseObjective(X, T, mu, S, params_dict)

	#for i in range(r):
	#	for j in range(r):
	#		print(f'{i,j} {logDelta[i,j]}')

	### performing the optimization
	prob = cp.Problem(cp.Minimize(objective), constraints)
	prob.solve(solver = params_dict['solver'], verbose = params_dict['printing'])#, save_file = 'dump.ptf')
	utils.tprint(f'solver exited with status {prob.status}')
	X_learned, T_learned, mu_learned = S@X.value, T.value, mu.value

	utils.tprint(f'mu_learned = {mu_learned}')
	### computing negativity
	if T.value is not None and params_dict['printing']:
		A = T.value*logDelta + np.conjugate(E.T)@np.reshape(FS@X.value, (R,R), order = 'F')@E
		negativity = float(min(scipy.linalg.eigh(A, eigvals_only = True)))
		utils.tprint(f'negativity of optimizer output: {negativity}')

	solver_stats = {}
	#solver_stats['compilation_time'] = prob.compilation_time
	solver_stats['status'] = prob.status
	solver_stats['T_learned'] = T.value
	solver_stats['extra_stats'] = prob.solver_stats.extra_stats
	solver_stats['num_iters'] = prob.solver_stats.num_iters
	solver_stats['solve_time'] = prob.solver_stats.solve_time
	solver_stats['solver_name'] = prob.solver_stats.solver_name
	if T.value is not None:
		solver_stats['negativity'] = negativity
	metrics['solver_stats'] = solver_stats

	#X_learned, T_learned, mu_learned = cvxSolve(logDelta, F_vectorized, W, hamiltonian_terms_expectations, params_dict, extras, metrics)

	if return_extras:
		extras = {}
		extras['twobody_operators'] = twobody_operators
		extras['threebody_operators'] = threebody_operators
		extras['mult_tensor'] = mult_tensor
		extras['triple_product_tensor'] = triple_product_tensor
		extras['C'] = C
		extras['F'] = F
		extras['E'] = E
		extras['W'] = W
		extras['h'] = h
		extras['dual_vector'] = constraints[-1].dual_value
		return np.concatenate(([1.],X_learned)), T_learned, extras
	else:
		return np.concatenate(([1.],X_learned)), T_learned

def learnHamiltonianFromThermalStateNew(n, onebody_operators, hamiltonian_terms, expectations_evaluator, params_dict, metrics, return_extras = False):
	### throughout, I assume that the identity is the first entry in hamiltonian_terms. It's used in the normalization
	assert hamiltonian_terms[0]=='I'*n
	hamiltonian_terms = hamiltonian_terms[1:]

	### building multiplication tensor
	if type(expectations_evaluator) == dict:
		expectations_evaluator = lambda x: expectations_evaluator[x]

	if params_dict['printing']:
		utils.tprint('building multiplication tensor')
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)

	if params_dict['printing']:
		utils.tprint('building threebody operators')
	threebody_operators = utils.buildThreeBodyTermsFast(onebody_operators, hamiltonian_terms, params_dict['printing'])

	if params_dict['printing']:
		utils.tprint('building triple product tensor')
	triple_product_tensor = utils.buildTripleProductTensorFast(onebody_operators, hamiltonian_terms, threebody_operators, params_dict['printing'])

	try:
		hamiltonian_terms_expectations = [expectations_evaluator(x) for x in hamiltonian_terms]
		twobody_expectations =[expectations_evaluator(x) for x in twobody_operators]
		threebody_expectations = [expectations_evaluator(x) for x in threebody_operators]
	except KeyError as inst:
		print(f'learnHamiltonianFromThermalstate was not given enough expectations. Missing operator: {str(inst)}')
		raise

	R = len(onebody_operators)
	h = len(hamiltonian_terms)

	### computing F tensor
	if params_dict['printing']:
		utils.tprint('building F tensor')

	triple_product_tensor = triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	F = triple_product_tensor.contractRight(threebody_expectations)#F_ijk = <a_i[h_k,a_j]> (note the order of indices)
	F_vectorized = F.vectorize(axes = [0,1]).toScipySparse()

	### computing covariance matrix C and modular operator Delta
	if params_dict['printing']:
		utils.tprint('computing covariance matrix (C) and modular operator (Delta)')
	
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()#C_ij = <a_ia_j>
	C_eigvals, C_eigvecs = scipy.linalg.eigh(C)

	# compute how many of the lowest eigenvalues of C to throw out
	C_eigval_cutoff = R
	for i in range(R):
		if C_eigvals[i] > params_dict['C_eigval_threshold']:
			C_eigval_cutoff = i
			break
	if C_eigval_cutoff == R:
		raise RuntimeError(f"no C eigenvalues were above threshold {params_dict['C_eigval_threshold']}")

	metrics['C_eigval_cutoff'] = C_eigval_cutoff
	if params_dict['printing']:
		utils.tprint(f'threw out the lowest {C_eigval_cutoff} eigenvalues of C')

	r = R - C_eigval_cutoff

	D = np.diag(np.reciprocal(np.sqrt(C_eigvals[C_eigval_cutoff:]))) #D = C^{-1/2}
	E = C_eigvecs[:,C_eigval_cutoff:]@D #E is D acting on the orthocomplement of the (approximate) kernel of C
	Delta = np.conjugate(E.T)@C.T@E
	
	Delta_eigs = scipy.linalg.eigvalsh(Delta)
	small = 1e-9 
	very_small = 1e-11
	set_T_zero_flag = False
	if Delta_eigs[0] < very_small:
		utils.tprint('Modular operator is singular. Setting T to zero and objective to l1 ')
		set_T_zero_flag = True
	elif Delta_eigs[0] < small:
		utils.tprint('WARNING: Modular operator is near singular')
	logDelta = scipy.linalg.logm(Delta)
	#TODO: just project onto Ker(Delta)^perp. In fact this should be automatic anyway
	#utils.plotSpec(logDelta, names = ['logDelta'], print_lowest = 10)

	### building W matrix
	utils.tprint('creating F_dag')
	F_dag = F.conjugate().transpose([1,0,2])
	utils.tprint('subtracting F from F_dag')
	Z = F - F_dag
	utils.tprint('vectorizing Z')
	Z_vec = Z.vectorizeLowerTriangular([0,1])
	utils.tprint('converting Z to scipy sparse')
	Z_vec_scipy = Z_vec.toScipySparse()
	utils.tprint('multiplying Z dag with Z')
	W = (Z_vec_scipy.T.conj()@Z_vec_scipy)
	utils.tprint('converting W to dense array')
	W = W.toarray()

	## computing S, whose columns form an orthonormal basis of the (approximate) null space of W
	W_eigvals, W_eigvecs = scipy.linalg.eigh(W)
	W_eigval_threshold = params_dict['W_eigval_threshold']
	q = list((W_eigvals > W_eigval_threshold)).index(True)
	if params_dict['printing']:
		utils.tprint('lowest 10 eigenvalues of W:')
		for i in range(10):
			utils.tprint(f' {W_eigvals[i]:.4e}')
	if q < 2 :
		raise ValueError(f'all W eigenvalues are above the cutoff {W_eigval_threshold}(except the trivial one coming from the identity)')
	utils.tprint(f'W cutoff = {q}')
	S = W_eigvecs[:,:q] 

	#TODO: check validity of this. It seems like W is always real and symmetric
	#imag(S) is exactly zero (no roundoff errors), even with noise in the expectations. So probably a good reason
	assert np.array_equal(np.imag(S), np.zeros(S.shape))
	S = np.real(S)
	s = F.shape[2]

	F_real = F.complexToReal([0,1])
	F_real_vectorized = F_real.vectorize([0,1]).toScipySparse()
	if not tuple(F_real.shape) == (2*r,2*r,s):
		raise ValueError(f'F_real has incorrect shape {F_real.shape} (expected {(2*r,2*r,s)})')
	#FS = np.zeros((2*r,2*r,q)) #dense rank-3 tensor. (FS)_ijk = (tilde{H}_k)_{ij}
	#for i in range(q):
	#	FS[:,:,i] = F_real.contractRight(S[:,i]).toNumpy()
	FS = (F_real_vectorized@S).reshape((2*r,2*r,q), order = 'F')

	omega_tilde = hamiltonian_terms_expectations@S

	###----------the convex optimization happens here----------###

	if params_dict['printing']:
		utils.tprint('learning Hamiltonian')

	with Model() as M:
		L = M.variable("L", Domain.inPSDCone(2*r))
		nu = M.variable("nu")

		#the 2r x 2r real matrix J represents multiplication by i
		#J = Matrix.sparse([[None, Expr.mul(-1,Matrix.eye(r))],[Matrix.eye(r),None]])
		indeces = np.vstack((np.hstack((np.arange(r),r + np.arange(r))), np.hstack((r + np.arange(r),np.arange(r)))))
		vals = np.concatenate((-np.ones(r), np.ones(r)))
		indeces = np.array(indeces, dtype = 'int32')
		vals = np.array(vals, dtype = 'double')
		
		J = Matrix.sparse(2*r,2*r, indeces[0], indeces[1], vals)
		M.objective(ObjectiveSense.Maximize, Expr.mul(-1,nu))

		#JL-LJ = 0, ie L is complex-linear
		#M.constraint(Expr.sub(Expr.mul(L,J), Expr.mul(J,L)), Domain.equalsTo(0))

		#tr(tilde{H}_alpha Lambda) + nu*omega(tilde{h}_alpha) = 0
		E_real = utils.complexToReal(E)
		expr = Expr.vstack([Expr.add(Expr.dot(E_real.T@FS[:,:,i]@E_real,L), Expr.mul(nu, omega_tilde[i])) for i in range(q)])
		M.constraint('tilde_h_alpha', expr , Domain.equalsTo(0))

		#tr(logDelta Lambda) \le 0
		M.constraint("logDelta_dot", Expr.dot(utils.complexToReal(logDelta), L), Domain.lessThan(0))

		#tr(Lambda) = 1
		M.constraint("trace_one", Expr.mul(1,Expr.dot(Matrix.eye(2*r),L)), Domain.equalsTo(1))

		M.setLogHandler(sys.stdout)
		M.setSolverParam("log",10000)
		#M.setSolverParam("infeasReportAuto", "on")
		
		#M.dataReport()
		M.solve()
		#M.writeTask('dump.ptf')

		X_value = S@M.getConstraint("tilde_h_alpha").dual()
		mu_value = M.getConstraint("trace_one").dual()[0]
		T_value = M.getConstraint("logDelta_dot").dual()[0]

		#print(X_value)
		utils.tprint(f'mu_value = {mu_value}')
		#print(f'T_value = {T_value}')

	#X_value, mu_value, T_value, solver_stats = mosekSolve(logDelta, F, W)

	if return_extras:
		extras = {}
		extras['twobody_operators'] = twobody_operators
		extras['threebody_operators'] = threebody_operators
		extras['mult_tensor'] = mult_tensor
		extras['triple_product_tensor'] = triple_product_tensor
		extras['C'] = C
		extras['F'] = F
		extras['E'] = E
		extras['W'] = W
		extras['h'] = h
		extras['mu_value'] = mu_value
		#extras['dual_vector'] = constraints[-1].dual_value
		return np.concatenate(([1.],X_value)), T_value, extras
	else:
		return np.concatenate(([1.],X_value)), T_value


def saveHamiltonian(observables_list, coefficients_list, filename):
	with open(filename, 'w') as f:  
		f.write(','.join(['pauli', 'pauli_compressed' ,'coefficient'])+'\n')
		l = len(observables_list)
		assert len(coefficients_list) == l
		for i in range(l):
			f.write(','.join([observables_list[i], compressPauli(observables_list[i]), str(coefficients_list[i])])+'\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('state_filename')
	#parser.add_argument('output_filename')
	args = parser.parse_args()

	k = 2
	state_evaluator, n = utils.buildStateEvaluator(args.state_filename, state_type = 'wavefunction')
	onebody_operators = utils.buildKLocalPaulis1D(n,k, periodic_bc = False)

	params_dict = {}

	hamiltonian_coefficients = learnHamiltonian(onebody_operators,state_evaluator)

	saveHamiltonian(onebody_operators, hamiltonian_coefficients, './learned_hamiltonian.txt')


	