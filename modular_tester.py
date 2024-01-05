import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from simulation import Hamiltonian, EquilibriumState
import hamiltonian_learning
import utils

def generateOperators(n,params):
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
	if not os.path.exists('./hamiltonians/'):
		os.mkdir('./hamiltonians/')
	H = Hamiltonian(n, f"./hamiltonians/{hamiltonian_filename}.yml", coupling_constants_dict)
	params_string = ''.join([f'{g}={coupling_constants_dict[g]}' for g in coupling_constants_dict.keys()])
	H_name = f'{n}_{hamiltonian_filename}_{params_string}'
	return H, H_name

#pads pauli strings with I's
def embedObservable(n, region, p):
	out = ['I']*n
	for i in range(len(region)):
		out[region[i]] = p[i]
	return ''.join(out)

#save_dir, metrics_and_params, H_in,H_learned)
def saveModularHamiltonianLearningResults(H_in, H_learned, T_learned, expectations_dict, region, params, metrics):
	if params['no_save']:
		save_dir = '.'
	else:
		save_dir = utils.createSaveDirectory()

	n = H_in.n
	m = len(region)

	'''
	if params['disorder'] > 0:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}_disordered.yml')
	else:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}.yml')
	'''
	H_in.saveToYAML(f"{save_dir}/{n}_{params['hamiltonian_filename']}.yml")
	H_learned.saveToYAML(f"{save_dir}/learned_hamiltonian.yml")

	with open(save_dir + '/metrics_and_params.yml', 'w') as f:
		yaml.dump(dict(metrics = metrics, params = params), f, default_flow_style=False)

	terms = H_learned.terms

	r = max([utils.weight(p) for p in terms])
	l = len(terms)

	learned_coeffs_dict = dict(zip(H_learned.terms,H_learned.coefficients))
	learned_normalization = H_learned.normalizeCoeffs(expectations_dict)

	#print(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}')
	if params['printing']:
		print()
		line = 'term' + ' '*(4*r-1) +':  mod ham coeff'
		print(line)
		print('-'*len(line))
		for p in terms:
			learned_coeff = learned_coeffs_dict[p]

			if np.abs(learned_coeff) > 1e-10:
				print(f'{utils.compressPauli(p)}' +
					' '*(4*r - len(utils.compressPauli(p)) + 2) +
					f' :  {learned_coeff:+.6f}')
		print()

	with open(save_dir + '/modular_tester_results.txt', 'w') as f:
		f.write('results of hamiltonian_learning_tester.py\n\n')
		if T_learned == None:
			f.write(f'learned T = {T_learned}\n\n')
		else:
			f.write(f'learned T = {T_learned:.10e}\n\n')
		#f.write(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}\n\n')
		line = 'term' + ' '*(4*r-1) +':  mod ham coeff \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for p in terms:
			learned_coeff = learned_coeffs_dict[p]

			f.write(f'{utils.compressPauli(p)}' +
					' '*(4*r - len(utils.compressPauli(p)) + 2) +
					f' :  {learned_coeff:+.10f}\n')

	if params['skip_plotting'] is False and T_learned is not None:
		if params['hamiltonian_filename'] == 'tf_ising_ferro':
			ZZ_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Z {k} Z {k+1}',m)] for k in range(len(region)-1)])
			X_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'1 X {k}',m)] for k in range(len(region))])

			normalization = 1/T_learned
			ZZ_learned_coeffs = normalization*ZZ_learned_coeffs
			X_learned_coeffs = normalization*X_learned_coeffs

			h = -params['coupling_constants']['g']
			k = min(1/abs(h),abs(h))
			k_prime = np.sqrt(1-k**2)
			if np.abs(h) < 1:
				##ordered phase
				X_theoretical_coeffs = 2*scipy.special.ellipk(k_prime)*np.flip([-k*(i + 0.5) for i in range(len(region))])
				ZZ_theoretical_coeffs = 2*scipy.special.ellipk(k_prime)*np.flip([-(i + 1) for i in range(len(region)-1)])
			else:
				##disordered phase
				X_theoretical_coeffs = 2*scipy.special.ellipk(k_prime)*np.flip([-(i + 0.5) for i in range(len(region))])
				ZZ_theoretical_coeffs = 2*scipy.special.ellipk(k_prime)*np.flip([-k*(i + 1) for i in range(len(region)-1)])

			plt.scatter(np.arange(len(region)), X_learned_coeffs, s=3, label = 'X coefficient')
			plt.scatter(np.arange(len(region)-1)+0.5, ZZ_learned_coeffs, s=3, label = 'ZZ coefficient')
			plt.plot(np.arange(len(region)), X_theoretical_coeffs, label = 'X coefficient (inf-volume theoretical)', linewidth = 1, alpha = 0.2)
			plt.plot(np.arange(len(region)-1), ZZ_theoretical_coeffs, label = 'ZZ coefficient (inf-volume theoretical)', linewidth = 1, alpha = 0.2)
			plt.xlabel('site')
			plt.title(f"n= {n} g = {params['coupling_constants']['g']} tfi modular hamiltonian on region {tuple(params['region'])}")
			plt.legend()
			plt.savefig(save_dir + "/reconstructed_hamiltonian.pdf", dpi=150)
			plt.show()

		elif params['hamiltonian_filename'] == 'xxz_Jneg':
			XX_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 X {k} X {k+1}',m)] for k in range(len(region)-1)])
			YY_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Y {k} Y {k+1}',m)] for k in range(len(region)-1)])
			ZZ_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Z {k} Z {k+1}',m)] for k in range(len(region)-1)])
			if XX_learned_coeffs[-1] > 1e-3:
				normalization = XX_learned_coeffs[-1]
			else:
				print('skipping normalizing because learned XX coefficient on the right is too small')
				normalization = 1
			XX_learned_coeffs = XX_learned_coeffs/normalization
			YY_learned_coeffs = YY_learned_coeffs/normalization
			ZZ_learned_coeffs = ZZ_learned_coeffs/normalization

			XX_theoretical_coeffs = [n-1-i for i in range(len(region)-1)]
			YY_theoretical_coeffs = [n-1-i for i in range(len(region)-1)]
			ZZ_theoretical_coeffs = [params['coupling_constants']['g']*(n-1-i) for i in range(len(region)-1)]

			colors = ['r','g','b']

			plt.plot(np.arange(len(region)-1), XX_theoretical_coeffs, c = 'r', label = 'XX coefficient (inf-volume theoretical)', alpha = 0.2)
			plt.plot(np.arange(len(region)-1), YY_theoretical_coeffs, c = 'g', label = 'YY coefficient (inf-volume theoretical)', alpha = 0.2)
			plt.plot(np.arange(len(region)-1), ZZ_theoretical_coeffs, c = 'b', label = 'ZZ coefficient (inf-volume theoretical)', alpha = 0.2)
			plt.scatter(np.arange(len(region)-1), XX_learned_coeffs, s=2, c = 'r', label = 'XX coefficient')
			plt.scatter(np.arange(len(region)-1), YY_learned_coeffs, s=2, c = 'g', label = 'YY coefficient')
			plt.scatter(np.arange(len(region)-1), ZZ_learned_coeffs, s=2, c = 'b', label = 'ZZ coefficient')
			plt.xlabel('site')
			plt.title(f"n= {n} g = {params['coupling_constants']['g']} xxz modular hamiltonian on region {tuple(params['region'])}")
			plt.legend()
			plt.savefig(save_dir + "/reconstructed_hamiltonian.pdf", dpi=150)
			plt.show()

		elif params['hamiltonian_filename'] == 'xxz':
			XX_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 X {k} X {k+1}',m)] for k in range(len(region)-1)])
			YY_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Y {k} Y {k+1}',m)] for k in range(len(region)-1)])
			ZZ_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Z {k} Z {k+1}',m)] for k in range(len(region)-1)])
			if XX_learned_coeffs[-1] > 1e-3:
				normalization = XX_learned_coeffs[-1]
			else:
				print('skipping normalizing because learned XX coefficient on the right is too small')
				normalization = 1
			XX_learned_coeffs = XX_learned_coeffs/normalization
			YY_learned_coeffs = YY_learned_coeffs/normalization
			ZZ_learned_coeffs = ZZ_learned_coeffs/normalization

			XX_theoretical_coeffs = [n-1-i for i in range(len(region)-1)]
			YY_theoretical_coeffs = [n-1-i for i in range(len(region)-1)]
			ZZ_theoretical_coeffs = [params['coupling_constants']['g']*(n-1-i) for i in range(len(region)-1)]

			colors = ['r','g','b']

			#plt.plot(np.arange(len(region)-1), XX_theoretical_coeffs, c = 'r', label = 'XX coefficient (inf-volume theoretical)', alpha = 0.2)
			#plt.plot(np.arange(len(region)-1), YY_theoretical_coeffs, c = 'g', label = 'YY coefficient (inf-volume theoretical)', alpha = 0.2)
			#plt.plot(np.arange(len(region)-1), ZZ_theoretical_coeffs, c = 'b', label = 'ZZ coefficient (inf-volume theoretical)', alpha = 0.2)
			plt.scatter(np.arange(len(region)-1), XX_learned_coeffs, s=2, c = 'r', label = 'XX coefficient')
			plt.scatter(np.arange(len(region)-1), YY_learned_coeffs, s=2, c = 'g', label = 'YY coefficient')
			plt.scatter(np.arange(len(region)-1), ZZ_learned_coeffs, s=2, c = 'b', label = 'ZZ coefficient')
			plt.xlabel('site')
			plt.title(f"n= {n} g = {params['coupling_constants']['g']} xxz modular hamiltonian on half-line")
			plt.legend()
			plt.savefig(save_dir + "/reconstructed_hamiltonian.pdf", dpi=150)
			plt.show()

if __name__ == '__main__':

	##load params dictionary
	assert len(sys.argv)==2
	setup_filename = sys.argv[1]
	with open(setup_filename) as f:
		params = yaml.safe_load(f)

	n = params['n']
	region = range(*tuple(params['region']))
	#region = range(n//2)
	m = len(region)

	### load Hamiltonian and give it a descriptive name
	H, H_name = loadAndNameHamiltonian(params)

	### generate operators used in convex optimization
	onebody_operators, hamiltonian_terms, threebody_operators = generateOperators(len(region),params)

	### generate embedded observables
	onebody_operators_embedded = [embedObservable(n,region,p) for p in onebody_operators]
	hamiltonian_terms_embedded = [embedObservable(n,region,p) for p in hamiltonian_terms]
	threebody_operators_embedded = [embedObservable(n,region,p) for p in threebody_operators]

	### initialize state
	if type(params['beta']) == float or type(params['beta']) == int:
		beta = params['beta']
	elif params['beta'] == 'inf':
		beta = np.inf
	else:
		raise ValueError
	state = EquilibriumState(params['n'],H,H_name,beta)

	###compute all required expectation values
	threebody_expectations =np.flip(state.getExpectations(np.flip(threebody_operators_embedded), params)) #we flip because getExpectations needs reverse order
	#threebody_expectations_ED = state.computeExpectationsED(threebody_operators_embedded)
	#for i in range(len(threebody_operators)):
	#	if np.abs(threebody_expectations[i]-threebody_expectations_ED[i]) > 1e-10:
	#		print(f'{threebody_operators[i]}  {threebody_expectations[i]}  {threebody_expectations_ED[i]}')
	expectations_dict = dict(zip(threebody_operators, threebody_expectations))
	expectations_embedded_dict = dict(zip(threebody_operators_embedded, threebody_expectations))

	### run convex optimization
	evaluator = lambda x : expectations_embedded_dict[x]
	hamiltonian_learned_coefficients, T_learned, C, F = hamiltonian_learning.learnHamiltonianFromThermalState(m, 
		onebody_operators_embedded, hamiltonian_terms_embedded, evaluator, params, state.metrics)
	if hamiltonian_learned_coefficients is not None:
		H_learned = Hamiltonian(len(region), hamiltonian_terms, hamiltonian_learned_coefficients)
	else:
		H_learned =  Hamiltonian(len(region), hamiltonian_terms, np.zeros(len(hamiltonian_terms)))

	### save results
	saveModularHamiltonianLearningResults(H, H_learned, T_learned, expectations_dict, region, params, state.metrics)


