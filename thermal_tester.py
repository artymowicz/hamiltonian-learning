from simulation import Hamiltonian, Simulator
import hamiltonian_learning
import utils


import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import yaml
import argparse

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

	utils.tprint(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}')
	utils.tprint(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}')
	if params['printing']:
		if params['objective'] == 'l2':
			utils.tprint(f"l2 norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = 2)}")
			utils.tprint(f"l2 norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = 2)}")
		else:
			utils.tprint(f"l1 norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = 1)}")
			utils.tprint(f"l1 norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = 1)}")
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
		f.write(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}\n\n')
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('setup_filename')
	parser.add_argument('-g', type = float)
	parser.add_argument('-b', '--beta', type = float)
	parser.add_argument('-dt', '--simulator_dt', type = float)
	parser.add_argument('-o', '--objective')
	parser.add_argument('-n', type = int)
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
	if args.objective is not None:
		params['n'] = args.n

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

	'''
	if state.n<10:
		params['simulator_method'] = 'ED'
		threebody_expectations_ED = state.getExpectations(threebody_operators, params)
		utils.tprint(f'MSE MPS vs ED: {np.square(np.linalg.norm(threebody_expectations_ED-threebody_expectations))/len(threebody_operators)}')
		#for i in range(len(threebody_operators)):
		#	print(f'{threebody_operators[i]}  {threebody_expectations[i]:.4f} {threebody_expectations_ED[i]:.4f} {threebody_expectations_naive[i]:.4f}')
	'''
	threebody_expectations_dict = dict(zip(threebody_operators,threebody_expectations))
	evaluator = lambda x : threebody_expectations_dict[x]
	### run convex optimization
	hamiltonian_learned_coefficients, T_learned, C, F = hamiltonian_learning.learnHamiltonianFromThermalState(params['n'], onebody_operators, hamiltonian_terms, evaluator, params, state.metrics)
	hamiltonian_learned_coefficients = hamiltonian_learned_coefficients

	H_learned = Hamiltonian(params['n'], hamiltonian_terms, hamiltonian_learned_coefficients)
	H_in_normalization = H.normalizeCoeffs(threebody_expectations_dict)


	### save results
	saveLearningResults(H, H_name, 1/params['beta']/H_in_normalization, H_learned, T_learned, threebody_expectations_dict, params, state.metrics)


