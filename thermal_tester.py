from simulation import Hamiltonian, EquilibriumState
import hamiltonian_learning
import utils

import sys
import numpy as np
import yaml

def saveLearningResults(H_in, H_in_name, T_in, H_learned, T_learned, expectations_dict, params, metrics):
	if params['no_save']:
		save_dir = '.'
	else:
		save_dir = utils.createSaveDirectory()

	n = H_in.n

	'''
	if params['disorder'] > 0:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}_disordered.yml')
	else:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}.yml')
	'''
	H_in.saveToYAML(f"{save_dir}/{n}_{params['hamiltonian_filename']}.yml")

	with open(save_dir + '/metrics_and_params.yml', 'w') as f:
		yaml.dump(dict(metrics = metrics, params = params), f, default_flow_style=False)

	terms = H_learned.terms

	r = max([utils.weight(p) for p in terms])
	l = len(terms)

	H_in.addTerms(['I'*n], [1])
	in_normalization = H_in.normalizeCoeffs(expectations_dict)
	learned_normalization = H_learned.normalizeCoeffs(expectations_dict)

	T_in = T_in/in_normalization
	T_learned = T_learned/learned_normalization

	in_coeffs_dict = dict(zip(H_in.terms,H_in.coefficients))
	learned_coeffs_dict = dict(zip(H_learned.terms,H_learned.coefficients))

	learned_coeffs = np.asarray([learned_coeffs_dict[p] for p in terms])
	in_coeffs = np.zeros(len(terms))
	for i in range(len(terms)):
		p = terms[i]
		if p in in_coeffs_dict:
			in_coeffs[i] = in_coeffs_dict[p]

	utils.tprint(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}')
	if params['printing']:
		utils.tprint(f"norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = params['objective_order'])}")
		utils.tprint(f"norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = params['objective_order'])}")
		print()
		line = 'term' + ' '*(4*r-1) +':  orig coeff :  learned coeff'
		print(line)
		print('-'*len(line))
		for i in range(len(terms)):
				if max(np.abs(in_coeffs[i]),np.abs(learned_coeffs[i])) > 1e-10:
					print(f'{utils.compressPauli(terms[i])}' +
						' '*(4*r - len(utils.compressPauli(terms[i])) + 2) +
						f' :  {in_coeffs[i]:+.6f}  :  {learned_coeffs[i]:+.6f}')
		print()

	with open(save_dir + f'/groundstate_learning_results.txt', 'w') as f:
		f.write(f'results of hamiltonian_learning_from_groundstate_tester.py\n\n')
		f.write(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}\n\n')
		f.write(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}\n\n')
		f.write(f"norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = params['objective_order'])}\n")
		f.write(f"norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = params['objective_order'])}\n\n")
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
	onebody_operators = utils.buildKLocalPaulis1D(n, k, periodic_bc)
	utils.tprint('computing Hamiltonian terms')
	if params['H_locality'] == 'short_range':
		hamiltonian_terms = utils.buildKLocalPaulis1D(n, k, periodic_bc)
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

	##load params dictionary
	assert len(sys.argv)==2
	setup_filename = sys.argv[1]
	with open(setup_filename) as f:
		params = yaml.safe_load(f)

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
	state = EquilibriumState(params['n'],H,H_name,beta)

	###compute all required expectation values
	threebody_expectations_dict = state.getExpectations(threebody_operators, params)

	### run convex optimization
	evaluator = lambda x : threebody_expectations_dict[x]
	hamiltonian_learned_coefficients, T_learned, C, F = hamiltonian_learning.learnHamiltonianFromThermalState(params['n'], onebody_operators, hamiltonian_terms, evaluator, params, state.metrics)
	H_learned = Hamiltonian(params['n'], hamiltonian_terms, hamiltonian_learned_coefficients)

	### save results
	saveLearningResults(H, H_name, 1/params['beta'], H_learned, T_learned, threebody_expectations_dict, params, state.metrics)


