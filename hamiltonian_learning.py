import numpy as np
import scipy
import cvxpy as cp
import argparse
import utils
import matplotlib.pyplot as plt

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
	triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	F = triple_product_tensor.contractRight(threebody_expectations)#F_ijk = <a_i[a_k,a_j]> (note the order of indices)
	F.vectorize(axes = [0,1])
	F_vectorized = F.toScipySparse()

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


'''
params_dict required arguments:

threshold: the threshold for computing the inverse of covariance matrix
mu: a small offset added to the positivity of the free energy
'''

#def learnHamiltonianFromThermalState(n,onebody_operators, hamiltonian_terms, expectations_evaluator, params_dict):
def learnHamiltonianFromThermalState(n, onebody_operators, hamiltonian_terms, expectations_evaluator, params_dict, metrics, return_extras = False):
	assert hamiltonian_terms[0]=='I'*n

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

	if params_dict['printing']:
		utils.tprint('building F tensor')

	triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	F = triple_product_tensor.contractRight(threebody_expectations)#F_ijk = <a_i[a_k,a_j]> (note the order of indices)
	F.vectorize(axes = [0,1])
	F_vectorized = F.toScipySparse()

	if params_dict['printing']:
		utils.tprint('building (thresholded) covariance matrix')

	if 'threshold' in params_dict:
		threshold = params_dict['threshold']
	else:
		utils.tprint(f'learnHamiltonianFromThermalState warning: threshold not found in params_dict. Setting threshold to 0')
		threshold = 0

	r = len(onebody_operators)
	h = len(hamiltonian_terms)
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()#C_ij = <a_ia_j>
	eigvals,eigvecs = scipy.linalg.eigh(C)
	cutoff = 0
	for i in range(r):
		if eigvals[i] > threshold:
			cutoff = i
			break
	metrics['C_eigval_cutoff'] = cutoff
	if params_dict['printing']:
		utils.tprint(f'cutoff = {cutoff}')

	D = np.diag(np.reciprocal(np.sqrt(eigvals[cutoff:])))
	E = eigvecs[:,cutoff:]@D

	D_inv = np.diag(np.sqrt(eigvals[cutoff:]))

	Delta = np.conjugate(E.T)@C.T@E
	logDelta = scipy.linalg.logm(Delta)

	if params_dict['printing']:
		utils.tprint('learning Hamiltonian')

	X = cp.Variable(h)
	T = cp.Variable()

	if params_dict['objective'] == 'l2':
		objective = cp.square(cp.norm(X,2))
	elif params_dict['objective'] == 'l1':
		objective = cp.norm(X,1)
	elif params_dict['objective'] == 'rand_linear':
		objective = X@np.random.normal(size = h)
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
	else:
		raise ValueError(f"objective {params_dict['objective']} not recognized. Valid inputs are 'l1', 'l2, 'rand_linear', 'T', or 'minus_T'")

	if params_dict['T_constraint'] == "T>0":
		constraints = [T>=0, X[0] == 1]
		constraints += [X@hamiltonian_terms_expectations == 0]
	elif params_dict['T_constraint'] == 'T=1':
		constraints = [T==1, X[0] == 0]
	else:
		raise ValueError(f"T_constraint {params_dict['T_constraint']} not recognized. Valid inputs are 'T=1' or 'T>0'.")

	constraints += [T*logDelta + np.conjugate(E.T)@cp.reshape(F_vectorized@X, (r,r))@E >> 0]

	#rootC = scipy.linalg.sqrtm(C)
	#g = lambda T, logDelta, E, F_vectorized, X : -T*rootC@scipy.linalg.logm(rootC@scipy.linalg.inv(C.T)@rootC)@rootC + cp.reshape(F_vectorized@X, (r,r))
	#constraints += [g(T, logDelta, E, F_vectorized, X) >> -params_dict['mu']]
	#constraints += [T*D_inv@logDelta@D_inv + np.conjugate(eigvecs[:,cutoff:].T)@cp.reshape(F_vectorized@X, (r,r))@eigvecs[:,cutoff:] >> 0 ]
	#g = 0#np.random.normal(size = (l,))
	#objective = cp.sum(cp.square(g-X))

	prob = cp.Problem(cp.Minimize(objective), constraints)
	prob.solve(solver = params_dict['solver'], verbose = params_dict['printing'])#, save_file = 'dump.ptf')
	#prob.solve(solver = 'SCS', verbose = True)#, save_file = 'dump.ptf')

	utils.tprint(f'solver exited with status {prob.status}')

	##negativity
	if T.value is not None:
		A = T.value*logDelta + np.conjugate(E.T)@np.reshape(F_vectorized@X.value, (r,r), order = 'F')@E
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

	if return_extras:

		extras = {}
		extras['twobody_operators'] = twobody_operators
		extras['threebody_operators'] = threebody_operators
		extras['mult_tensor'] = mult_tensor
		extras['triple_product_tensor'] = triple_product_tensor
		extras['C'] = C
		extras['F'] = F
		extras['dual_vector'] = constraints[-1].dual_value
		extras['E'] = E

		
		return X.value, T.value, extras
	else:
		return X.value, T.value

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


	