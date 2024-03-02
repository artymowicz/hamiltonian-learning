from simulation import Hamiltonian, Simulator
import hamiltonian_learning
import utils

import os
import pickle
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import yaml
import argparse
from matplotlib import colormaps


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

def dag(A):
	return np.conjugate(A.T)

def eigvecs(A):
	return scipy.linalg.eigh(A)[1]

def analysis(params, data):
	onebody_operators = data['onebody_operators']
	twobody_operators = data['twobody_operators']
	hamiltonian_terms = data['hamiltonian_terms']
	threebody_operators = data['threebody_operators']
	expectations_dict = data['expectations_dict']
	mult_tensor = data['mult_tensor']
	triple_product_tensor = data['triple_product_tensor']
	dual_vector = data['dual_vector']
	hamiltonian_learned_coefficients = data['hamiltonian_learned_coefficients']
	H_in = data['H_in']
	T_in_normalized = data['T_in_normalized']
	T_learned = data['T_learned']
	F = data['F']

	E = data['E'] # = C^-1/2
	C = data['C']
	n = params['n']
	R = len(onebody_operators)
	r = E.shape[0]

	Delta = np.conjugate(E.T)@C.T@E
	logDelta = scipy.linalg.logm(Delta)
	#utils.plotSpec(Delta, title = 'Delta spectrum')#, yscale = 'log')

	F_vectorized = F.vectorize([0,1]).toScipySparse()
	print(f'T_learned = {T_learned}')
	print(f"T_theor = {T_in_normalized}")

	hamiltonian_theoretical_coefficients = np.zeros(len(hamiltonian_terms))
	for i in range(len(H_in.terms)):
		j = hamiltonian_terms.index(H_in.terms[i])
		hamiltonian_theoretical_coefficients[j] = H_in.coefficients[i]

	free_energy_learned = T_learned*logDelta + np.conjugate(E.T)@np.reshape(F_vectorized@hamiltonian_learned_coefficients, (r,r), order = 'F')@E
	free_energy_theoretical = T_in_normalized*logDelta + np.conjugate(E.T)@np.reshape(F_vectorized@hamiltonian_theoretical_coefficients, (r,r), order = 'F')@E

	#utils.plotSpec(dual_vector, title = 'dual vector spectrum', yscale = 'log')
	#utils.plotSpec(free_energy_learned, free_energy_theoretical, names = ['learned', 'theoretical'] , title = 'free energy spectra' )

	utils.tprint('plotting')
	if not params_dict['skip_plotting']:
		utils.plotSpec(Delta, C,W, names = ['Delta', 'C','W'], yscale = 'log', print_lowest = 10)

	utils.plotSpec(free_energy_theoretical, names = ['theoretical'] , title = 'free energy spectra')

	'''
	utils.plotSpec(Delta)
	Delta_eigvals, Delta_eigvecs = scipy.linalg.eigh(Delta)
	v0 = E@Delta_eigvecs[:,0]
	v1 = E@Delta_eigvecs[:,1]
	print(Delta_eigvals[:10])
	for i in range(len(onebody_operators)):
		print(f'{onebody_operators[i]}  {v0[i]}  {v1[i]}')
	'''

	'''
	assert n > 1
	p = 'XX' + 'I'*(n-2)
	i = twobody_operators.index(p)
	v = np.zeros(len(twobody_operators))
	v[i] = 1
	C_prime = mult_tensor.contractRight(v).toNumpy()

	#wont work when cutoff > 0
	#Delta_prime = np.conjugate(derivInvSqrt(C,C_prime).T)@C.T@E + np.conjugate(E.T)@C_prime.T@E + np.conjugate(E.T)@C_prime.T@derivInvSqrt(C,C_prime)

	Delta_eigvals = scipy.linalg.eigvalsh(Delta)
	thresh = 1e-10
	nonzero_indices = np.array([i for i in range(r) if np.abs(Delta_eigvals[i] - 1) > thresh])
	zero_indices = np.array([i for i in range(r) if i not in (nonzero_indices)])
	P = eigvecs(C)[:,nonzero_indices] ## projection onto nonzero eigenspace of Delta
	P_perp = eigvecs(C)[:,zero_indices] ## projection onto zero eigenspace of Delta
	A = dag(P)@Delta_prime@P
	B = dag(P_perp)@Delta_prime@P_perp
	utils.plotSpec(A, title = f'A spectrum for {p}')
	utils.plotSpec(B, title = f'B spectrum for {p}')
	'''

	#Delta_prime = np.conjugate(derivInvSqrt(C,C_prime).T)@C.T@E + np.conjugate(E.T)@C_prime.T@E + np.conjugate(E.T)@C_prime.T@derivInvSqrt(C,C_prime)
	#logDelta_prime = derivLog(Delta,Delta_prime)

	#utils.plotSpec(logDelta_prime, title = 'logDeltaPrime spectrum')

	#print(f'lam dot logDelta_prime = {np.trace(dag(dual_vector)@logDelta_prime)}')

	#Delta = np.conjugate(E.T)@C.T@E
	#logDelta = scipy.linalg.logm(Delta)

def saveLearningResults(H_in, H_in_name, T_in, H_learned, T_learned, expectations_dict, params, metrics):
	if params['no_save']:
		save_dir = '.'
	else:
		save_dir = utils.createSaveDirectory()

	n = H_in.n

	'''
	if params['disorder'] > 0:
		H_in.saveToYAML(f"{save_dir}/{n}_{params['hamiltonian_filename']}_disordered.yml")
	else:
		H_in.saveToYAML(f"{save_dir}/{n}_{params['hamiltonian_filename']}.yml")
	'''
	H_in.saveToYAML(f"{save_dir}/{n}_{params['hamiltonian_filename']}.yml")

	with open(save_dir + '/metrics_and_params.yml', 'w') as f:
		yaml.dump(dict(metrics = metrics, params = params), f, default_flow_style=False)

	terms = H_learned.terms

	r = max([utils.weight(p) for p in terms])
	l = len(terms)

	#H_in.addTerms(['I'*n], [1])
	#in_normalization = H_in.normalizeCoeffs(expectations_dict)
	#learned_normalization = H_learned.normalizeCoeffs(expectations_dict)

	#T_in = T_in/in_normalization
	#T_learned = T_learned/learned_normalization

	in_coeffs_dict = dict(zip(H_in.terms,H_in.coefficients))
	learned_coeffs_dict = dict(zip(H_learned.terms,H_learned.coefficients))

	learned_coeffs = np.asarray([learned_coeffs_dict[p] for p in terms])
	in_coeffs = np.zeros(len(terms))
	for i in range(len(terms)):
		p = terms[i]
		if p in in_coeffs_dict:
			in_coeffs[i] = in_coeffs_dict[p]

	def angle(x,y):
		n_x = np.linalg.norm(x)
		n_y = np.linalg.norm(y)
		overlap = np.abs(np.vdot(x,y))
		return np.arccos(min(overlap/(n_x*n_y),1.))

	theta = angle(in_coeffs, learned_coeffs)

	utils.tprint(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}')
	utils.tprint(f'reconstruction error (theta) is {theta}')
	if params['printing']:
		if params['objective'] == 'l2':
			utils.tprint(f"l2 norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = 2)}")
			utils.tprint(f"l2 norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = 2)}")

		print()
		line = 'term' + ' '*(4*r-1) +':  orig coeff :  learned coeff'
		print(line)
		print('-'*len(line))
		for i in range(len(terms)):
				if max(np.abs(in_coeffs[i]),np.abs(learned_coeffs[i])) > 1e-8:
					print(f'{utils.compressPauli(terms[i])}' +
						' '*(4*r - len(utils.compressPauli(terms[i])) + 2) +
						f' :  {in_coeffs[i]:+.8f}  :  {learned_coeffs[i]:+.8f}')
		print()

	with open(save_dir + f'/groundstate_learning_results.txt', 'w') as f:
		f.write(f'results of hamiltonian_learning_from_groundstate_tester.py\n\n')
		f.write(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}\n\n')
		f.write(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}\n')
		f.write(f'reconstruction error (theta) is {theta}\n\n')

		if params['objective'] == 'l2':
			f.write(f"l2 norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = 2)}\n")
			f.write(f"l2 norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = 2)}\n")
		else:
			f.write(f"l1 norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = 1)}\n")
			f.write(f"l1 norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = 1)}\n")
		line = 'term' + ' '*(4*r-1) +':  orig coeff     :  learned coeff \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for i in range(l):
			f.write(f'{utils.compressPauli(terms[i])}' +
					' '*(4*r - len(utils.compressPauli(terms[i])) + 2) +
					f' :  {in_coeffs[i]:+.10f}  :  {learned_coeffs[i]:+.10f}\n')

	return save_dir

def generateOperators(params):
	n = params['n']
	k = params['k']
	periodic_bc = params['periodic']
	utils.tprint('computing onebody operators')
	if params['onebody_locality'] == 'short_range':
		onebody_operators = utils.buildKLocalPaulis1D(n, k, periodic_bc)
	elif params['onebody_locality'] == 'long_range':
		onebody_operators = utils.buildKLocalCorrelators1D(n,params['k'], periodic_bc, d_max = params['onebody_range'])
	else:
		raise ValueError(f"unrecognized value for onebody_locality: {params['onebody_locality']}")
	
	utils.tprint('computing Hamiltonian terms')
	if params['H_locality'] == 'short_range':
		hamiltonian_terms = utils.buildKLocalPaulis1D(n, params['k_H'], periodic_bc)
	elif params['H_locality'] == 'long_range':
		hamiltonian_terms = utils.buildKLocalCorrelators1D(n,params['k_H'], periodic_bc, d_max = params['H_range'])
	else:
		raise ValueError(f"unrecognized value for H_locality: {params['H_locality']}")

	utils.tprint('computing three-body terms')
	threebody_operators = utils.buildThreeBodyTermsFast(onebody_operators, hamiltonian_terms, params['printing'])
	utils.tprint(f'n: {n} number of three-body terms: {len(threebody_operators)}')

	return onebody_operators, hamiltonian_terms, threebody_operators

def loadAndNameHamiltonian(params):
	n = params['n']
	hamiltonian_filename = params['hamiltonian_filename']
	coupling_constants_dict = params['coupling_constants']
	H = Hamiltonian(n, f"./hamiltonians/{hamiltonian_filename}.yml", coupling_constants_dict)
	params_string = ''.join([f'{g}={coupling_constants_dict[g]}' for g in coupling_constants_dict.keys()])
	H_name = f'{n}_{hamiltonian_filename}_{params_string}'
	return H, H_name

def simulateShadowTomographyNoise(operators, expectations, n_measurements):
	assert len(operators) == len(expectations)
	variances = np.zeros(len(operators))
	for i in range(len(operators)):
		weight = utils.weight(operators[i])
		if weight == 0:
			variances[i]=0
		else:
			variances[i] = (3**weight)*(1-expectations[i]**2)/n_measurements
	noise = np.sqrt(variances)*np.random.normal(size=len(operators))
	return noise

def main(params):
	utils.tprint(f'running modular_tester.py with params:')
	print(yaml.dump(params))

	### load Hamiltonian and give it a descriptive name
	H, H_name = loadAndNameHamiltonian(params)

	### generate operators used in convex optimization
	onebody_operators, hamiltonian_terms, threebody_operators = generateOperators(params)

	### initialize state
	if type(params['beta']) == float or type(params['beta']) == int:
		beta = params['beta']
	elif params['beta'] == 'inf':
		beta = np.inf
	else:
		raise ValueError
	state = Simulator(params['n'],H,H_name,beta)

	#threebody_expectations = np.flip(state.getExpectations(np.flip(threebody_operators), params))
	threebody_expectations = np.flip(state.getExpectations(np.flip(threebody_operators), params))
	#threebody_expectations += params['noise']*np.random.normal(size = len(threebody_operators))
	if params['add_noise']:
		utils.tprint('adding noise')
		if params['ST_measurements'] is not None:
			noise = simulateShadowTomographyNoise(threebody_operators, threebody_expectations, params['ST_measurements'])
			#for i in range(len(threebody_operators)):
			#	print(f'{threebody_operators[i]} {threebody_expectations[i]} {noise[i]}')
			threebody_expectations += noise
		if params['uniform_noise']:
			threebody_expectations += params['uniform_noise']*np.random.normal(size=len(threebody_expectations))
	'''
	if state.n<10:
		params['simulator_method'] = 'ED'
		threebody_expectations_ED = state.getExpectations(threebody_operators, params)
		utils.tprint(f'MSE MPS vs ED: {np.square(np.linalg.norm(threebody_expectations_ED-threebody_expectations))/len(threebody_operators)}')
		#for i in range(len(threebody_operators)):
		#	print(f'{threebody_operators[i]}  {threebody_expectations[i]:.4f} {threebody_expectations_ED[i]:.4f} {threebody_expectations_naive[i]:.4f}')
	'''
	utils.tprint('creating expectations dict')
	threebody_expectations_dict = dict(zip(threebody_operators,threebody_expectations))
	evaluator = lambda x : threebody_expectations_dict[x]

	### run convex optimization
	
	#assert onebody_operators[0] == 'I'*params['n']
	#onebody_operators = onebody_operators[1:]
	args = (params['n'], onebody_operators, hamiltonian_terms, evaluator, params, state.metrics)
	kwargs = dict(return_extras = True)
	hamiltonian_learned_coefficients, T_learned, run_data = hamiltonian_learning.learnHamiltonianFromThermalStateNew(*args, **kwargs)

	#for i in range(len(hamiltonian_terms)):
	#	print(f'{hamiltonian_terms[i]}  {hamiltonian_learned_coefficients[i]}')

	H_learned = Hamiltonian(params['n'], hamiltonian_terms, hamiltonian_learned_coefficients)
	H_in_normalization = H.normalizeCoeffs(threebody_expectations_dict)
	print(f'H_in_normalization = {H_in_normalization}')

	run_data['H_in'] = H
	run_data['T_in_normalized'] = 1/params['beta']/H_in_normalization
	run_data['onebody_operators'] = onebody_operators
	run_data['hamiltonian_terms'] = hamiltonian_terms
	run_data['expectations_dict'] = threebody_expectations_dict
	run_data['T_learned'] = T_learned
	run_data['hamiltonian_learned_coefficients'] = hamiltonian_learned_coefficients

	### save results
	saveLearningResults(H, H_name, run_data['T_in_normalized'], H_learned, T_learned, threebody_expectations_dict, params, state.metrics)

	return run_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('setup_filename')
	parser.add_argument('-g', type = float)
	parser.add_argument('-b', '--beta', type = float)
	parser.add_argument('-dt', '--simulator_dt', type = float)
	parser.add_argument('-mu', type = float)
	parser.add_argument('--unoise', type = float) #uniform noise
	parser.add_argument('-o', '--objective')
	parser.add_argument('-n', type = int)
	parser.add_argument('-wt','--w_thresh', type = float) #W_eigval_threshold
	parser.add_argument('-ct','--c_thresh', type = float) #C_eigval_threshold
	parser.add_argument('--st_meas', type = float)

	args = parser.parse_args()

	##load params dictionary
	with open(args.setup_filename) as f:
		params = yaml.safe_load(f)

	## some parameters can be provided in the commandline, they will then override the ones in the setup file
	if args.g is not None:
		params['coupling_constants']['g'] = args.g
	if args.beta is not None:
		params['beta'] = args.beta
	if args.simulator_dt is not None:
		params['simulator_dt'] = args.simulator_dt
	if args.objective is not None:
		params['objective'] = args.objective
	if args.n is not None:
		params['n'] = args.n
	if args.mu is not None:
		params['mu'] = args.mu
	if args.w_thresh is not None:
		params['W_eigval_threshold'] = args.w_thresh
	if args.c_thresh is not None:
		params['C_eigval_threshold'] = args.c_thresh
	if args.unoise is not None:
		params['uniform_noise'] = args.unoise
	if args.st_meas is not None:
		params['ST_measurements'] = args.st_meas

	use_cached = False #caching of previous run. Not intended for use on cluster
	if use_cached:
		loaded_successfully = False
		if os.path.exists('./caches/lastrun/params.yml'):
			with open('./caches/lastrun/params.yml', 'r') as f:
				params_cached = yaml.safe_load(f)
			if set(params_cached.keys()) & set(params.keys()) and all([params_cached[key] == params[key] for key in params.keys()]):
				try:
					with open('./caches/lastrun/data.pkl', 'rb') as f:
						data = pickle.load(f)
					loaded_successfully = True
				except Exception as e:
					print('encountered error trying to load last run: ')
					print(f'    {e}')

		if not loaded_successfully:
			data = main(params)
			if not os.path.exists('./caches/lastrun/'):
				os.mkdir('./caches/lastrun/')
			with open('./caches/lastrun/params.yml', 'w') as f:
				yaml.dump(params, f)
			with open('./caches/lastrun/data.pkl', 'wb') as f:
				pickle.dump(data, f)
	else:
		data = main(params)

	#analysis(params, data)

