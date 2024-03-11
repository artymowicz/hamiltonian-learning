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
import textwrap

def printLearningResults(run_data, params):
	T_in = run_data['T_in_normalized']
	T_learned = run_data['T_learned']
	n = params['n']
	terms = run_data['hamiltonian_terms']
	in_coeffs = run_data['hamiltonian_in_coefficients']
	learned_coeffs = run_data['hamiltonian_learned_coefficients']
	theta = run_data['theta']
	mu = run_data['mu']

	max_weight = max([utils.weight(p) for p in terms])
	l = len(terms)

	print()
	print(f'original T = {T_in:.10e}')
	print(f'learned T = {T_learned:.10e}')
	print(f'mu = {mu:.10e}')
	print(f'reconstruction error (theta) = {theta}')
	print()

	if params['printing_level']>3:
		line = 'term' + ' '*(4*max_weight-1) +':   orig coeff  :   learned coeff'
		print(line)
		print('-'*len(line))
		for i in range(len(terms)):
				if max(np.abs(in_coeffs[i]),np.abs(learned_coeffs[i])) > 1e-8:
					print(f'{utils.compressPauli(terms[i])}' +
						' '*(4*max_weight - len(utils.compressPauli(terms[i])) + 2) +
						f' :  {in_coeffs[i]:+.8f}  :  {learned_coeffs[i]:+.8f}')
		print()

def saveLearningResults(run_data, params):
	T_in = run_data['T_in_normalized']
	T_learned = run_data['T_learned']
	n = params['n']
	terms = run_data['hamiltonian_terms']
	in_coeffs = run_data['hamiltonian_in_coefficients']
	learned_coeffs = run_data['hamiltonian_learned_coefficients']
	theta = run_data['theta']
	H_in = Hamiltonian(n, terms, in_coeffs)
	H_learned = Hamiltonian(n, terms, learned_coeffs)

	if params['no_save']:
		save_dir = '.'
	else:
		save_dir = utils.createSaveDirectory()

	H_in.saveToYAML(f"{save_dir}/{n}_{params['hamiltonian_filename']}.yml")
	H_learned.saveToYAML(f"{save_dir}/learned_hamiltonian.yml")

	with open(save_dir + '/data.yml', 'w') as f:
		d = dict(metrics = run_data['metrics'], 
			params = params, 
			T_in_normalized = float(run_data['T_in_normalized']),
			T_in = float(run_data['T_in']),
			T_learned = float(run_data['T_learned']),
			mu = float(run_data['mu']),
			q = int(run_data['q']),
			theta = float(run_data['theta']))
		yaml.dump(d, f, default_flow_style=False)

	max_weight = max([utils.weight(p) for p in terms])
	l = len(terms)

	with open(save_dir + f'/results.txt', 'w') as f:
		f.write(f'results of tester.py\n\n')
		f.write(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}\n\n')
		f.write(f'reconstruction error (theta) is {theta}\n\n')

		line = 'term' + ' '*(4*max_weight-1) +':  orig coeff     :  learned coeff \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for i in range(l):
			f.write(f'{utils.compressPauli(terms[i])}' +
					' '*(4*max_weight - len(utils.compressPauli(terms[i])) + 2) +
					f' :  {in_coeffs[i]:+.10f}  :  {learned_coeffs[i]:+.10f}\n')

	return save_dir

def generateOperators(params):
	n = params['n']
	k = params['k']
	periodic_bc = params['periodic']
	if params['printing_level'] > 2:
		utils.tprint('computing onebody operators')
	if params['onebody_locality'] == 'short_range':
		onebody_operators = utils.buildKLocalPaulis1D(n, k, periodic_bc)
	elif params['onebody_locality'] == 'long_range':
		onebody_operators = utils.buildKLocalCorrelators1D(n,params['k'], periodic_bc, d_max = params['onebody_range'])
	else:
		raise ValueError(f"unrecognized value for onebody_locality: {params['onebody_locality']}")
	
	if params['printing_level'] > 2:
		utils.tprint('computing Hamiltonian terms')
	if params['H_locality'] == 'short_range':
		hamiltonian_terms = utils.buildKLocalPaulis1D(n, params['k_H'], periodic_bc)
	elif params['H_locality'] == 'long_range':
		hamiltonian_terms = utils.buildKLocalCorrelators1D(n,params['k_H'], periodic_bc, d_max = params['H_range'])
	else:
		raise ValueError(f"unrecognized value for H_locality: {params['H_locality']}")

	if params['printing_level'] > 2:
		utils.tprint('computing three-body terms')
	threebody_operators = utils.buildThreeBodyTerms(onebody_operators, hamiltonian_terms)

	if params['printing_level'] > 2:
		utils.tprint(f'number of three-body terms: {len(threebody_operators)}')

	### we do not want the identity to be a Hamiltonian term
	assert hamiltonian_terms[0]=='I'*n
	hamiltonian_terms = hamiltonian_terms[1:]

	return onebody_operators, hamiltonian_terms, threebody_operators

def loadAndNameHamiltonian(params):
	n = params['n']
	hamiltonian_filename = params['hamiltonian_filename']
	coupling_constants_dict = params['coupling_constants']
	H = Hamiltonian(n, f"./hamiltonians/{hamiltonian_filename}.yml", coupling_constants_dict)
	if np.abs(params['disorder']) > 0 :
		H.addDisorder(params['disorder'])
	params_string = '_'.join([f'{g}={coupling_constants_dict[g]}' for g in coupling_constants_dict.keys()])
	
	if np.abs(params['disorder']) > 0 :
		params_string = params_string + '_disordered'

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

def tester(params):
	### load Hamiltonian and generate a descriptive name
	H, H_name = loadAndNameHamiltonian(params)

	### generate lists of operators used in convex optimization
	onebody_operators, hamiltonian_terms, threebody_operators = generateOperators(params)

	### initialize state
	if type(params['beta']) == float or type(params['beta']) == int:
		beta = params['beta']
	elif params['beta'] == 'inf':
		beta = np.inf
	else:
		raise ValueError

	state = Simulator(params['n'],H,H_name,beta)

	### compute expectations and add noise if applicable
	threebody_expectations = state.getExpectations(threebody_operators, params)

	if params['add_noise']:
		if  params['printing_level'] > 2:
			utils.tprint('adding noise to expectation values')
		if params['ST_measurements'] != 0:
			noise = simulateShadowTomographyNoise(threebody_operators, threebody_expectations, params['ST_measurements'])
			threebody_expectations += noise
		if params['uniform_noise']:
			threebody_expectations += params['uniform_noise']*np.random.normal(size=len(threebody_expectations))

	expectations_dict = dict(zip(threebody_operators,threebody_expectations))

	if  params['printing_level'] > 2:
		utils.tprint('building multiplication tensor')
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)

	if  params['printing_level'] > 2:
		utils.tprint('building triple product tensor')
	triple_product_tensor = utils.buildTripleProductTensor(onebody_operators, hamiltonian_terms, threebody_operators)
	if  params['printing_level'] > 2:
		utils.tprint(f'triple product tensor has {len(triple_product_tensor.values)} nonzero values')

	try:
		hamiltonian_terms_expectations = [expectations_dict[x] for x in hamiltonian_terms]
		twobody_expectations = [expectations_dict[x] for x in twobody_operators]
	except KeyError as inst:
		print(f'not enough expectations to build F tensor. Missing operator: {str(inst)}')
		raise

	if params['printing_level'] > 2:
		utils.tprint('building F tensor')

	triple_product_tensor = triple_product_tensor.transpose([0,2,1,3])#we want the Hamiltonian index to be second-last
	F = triple_product_tensor.contractRight(threebody_expectations)#F_ijk = <b_i[h_k,b_j]> (note the order of indices)

	if params['printing_level'] > 2:
		utils.tprint(f'F tensor has {len(F.values)} nonzero entries')

	if  params['printing_level'] > 2:
		utils.tprint('computing covariance matrix C')
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()#C_ij = <b_ib_j>

	###----- we call the hamiltonian learning algorithm here -----###
	r = len(onebody_operators)
	s = len(hamiltonian_terms)
	J = np.eye(r, dtype = complex)

	### this formula for epsilon_W was determined empirically
	if params['add_noise']:
		epsilon_W = params['epsilon_W_prefactor']*max(np.sqrt(len(threebody_operators))*((params['uniform_noise'])**2), 1e-11)
	else:
		epsilon_W = params['epsilon_W_prefactor']*1e-11

	printing_level = params['printing_level']
	args = (r, s, hamiltonian_terms_expectations, J, C, F.indices, F.values, epsilon_W, printing_level)

	hamiltonian_learned_coefficients, T_learned, mu, q = hamiltonian_learning.learnHamiltonianFromThermalState(*args)

	if params['printing_level'] > 0:
		utils.tprint('Hamiltonian learning complete')

	### ----------------------------------------------------------###

	H_learned = Hamiltonian(params['n'], hamiltonian_terms, hamiltonian_learned_coefficients)
	H_in_normalization = H.normalizeCoeffs(expectations_dict)
	H_in_coeffs_dict = dict(zip(H.terms, H.coefficients))
	hamiltonian_in_coefficients = np.zeros(len(hamiltonian_terms))
	for i in range(len(hamiltonian_terms)):
		p = hamiltonian_terms[i]
		if p in H_in_coeffs_dict:
			hamiltonian_in_coefficients[i] = H_in_coeffs_dict[p]

	run_data = {}
	run_data['T_in'] = 1/params['beta']
	run_data['T_in_normalized'] = 1/params['beta']/H_in_normalization
	run_data['onebody_operators'] = onebody_operators
	run_data['hamiltonian_terms'] = hamiltonian_terms
	run_data['expectations_dict'] = expectations_dict
	run_data['T_learned'] = T_learned
	run_data['hamiltonian_in_coefficients'] = hamiltonian_in_coefficients
	run_data['hamiltonian_learned_coefficients'] = hamiltonian_learned_coefficients
	run_data['mu'] = mu
	run_data['q'] = q

	def angle(x,y):
		n_x = np.linalg.norm(x)
		n_y = np.linalg.norm(y)
		overlap = np.abs(np.vdot(x,y))
		return np.arccos(min(overlap/(n_x*n_y),1.))

	run_data['theta'] = angle(hamiltonian_in_coefficients, hamiltonian_learned_coefficients)
	run_data['metrics'] = state.metrics

	return run_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('setup_filename')
	parser.add_argument('-g', type = float)
	parser.add_argument('-b', '--beta', type = float)
	parser.add_argument('-dt', '--simulator_dt', type = float)
	parser.add_argument('--unoise', type = float)
	parser.add_argument('-n', type = int)
	parser.add_argument('--st_meas', type = float)
	parser.add_argument('--n_runs', type = int, default = 1)
	parser.add_argument('-d', '--disorder', type = float)
	parser.add_argument('-pl', '--printing_level', type = int)
	parser.add_argument('-k', type = int)
	parser.add_argument('--epsilon_W_prefactor', type = float)
	args = parser.parse_args()

	### load params dictionary
	with open(args.setup_filename) as f:
		params = yaml.safe_load(f)

	### some parameters can be provided in the commandline (they will override the ones in the setup file)
	if args.g is not None:
		params['coupling_constants']['g'] = args.g
	if args.beta is not None:
		params['beta'] = args.beta
	if args.simulator_dt is not None:
		params['simulator_dt'] = args.simulator_dt
	if args.n is not None:
		params['n'] = args.n
	if args.unoise is not None:
		params['uniform_noise'] = args.unoise
	if args.st_meas is not None:
		params['ST_measurements'] = args.st_meas
	if args.disorder is not None:
		params['disorder'] = args.disorder
	if args.printing_level is not None:
		params['printing_level'] = args.printing_level
	if args.k is not None:
		params['k'] = args.k
	if args.epsilon_W_prefactor is not None:
		params['epsilon_W_prefactor'] = args.epsilon_W_prefactor

	if  params['printing_level'] > 0:
		print()

	if params['printing_level'] > 2:
		utils.tprint(f'running tester.py with params:')
		print()
		print(textwrap.indent(yaml.dump(params), '    '))

	for i in range(args.n_runs):
		if args.n_runs > 1:
			utils.tprint(f'Run number {i+1} of {args.n_runs}')
		run_data = tester(params)
		if params['printing_level'] > 0:
			printLearningResults(run_data, params)
		saveLearningResults(run_data, params)

		### add in any analysis here
		#analysis(params, data)

