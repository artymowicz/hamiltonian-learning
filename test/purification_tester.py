import argparse
import state_simulator
import utils
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import os
import time
import yaml
import pickle

from state_simulator import Hamiltonian

from tenpy.models.tf_ising import TFIChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg

def imag_tebd(L=30, beta_max=3., dt=0.05, order=2, bc="finite"):
	model_params = dict(L=L, J=1., g=1.2)
	M = TFIChain(model_params)
	psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
	options = {
		'trunc_params': {
			'chi_max': 100,
			'svd_min': 1.e-8
		},
		'order': order,
		'dt': dt,
		'N_steps': 1
	}
	beta = 0.
	eng = PurificationTEBD(psi, M, options)
	Szs = [psi.expectation_value("Sz")]
	betas = [0.]
	while beta < beta_max:
		beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
		betas.append(beta)
		eng.run_imaginary(dt)  # cool down by dt
		Szs.append(psi.expectation_value("Sz"))  # and further measurements...
	return {'beta': betas, 'Sz': Szs}

def imag_apply_mpo(L=30, beta_max=3., dt=0.05, order=2, bc="finite", approx="II"):
	model_params = dict(L=L, J=1., g=1.2)
	M = TFIChain(model_params)
	psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
	options = {'trunc_params': {'chi_max': 100, 'svd_min': 1.e-8}}
	beta = 0.
	if order == 1:
		Us = [M.H_MPO.make_U(-dt, approx)]
	elif order == 2:
		Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
	eng = PurificationApplyMPO(psi, Us[0], options)
	Szs = [psi.expectation_value("Sz")]
	betas = [0.]
	while beta < beta_max:
		beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
		betas.append(beta)
		for U in Us:
			eng.init_env(U)  # reset environment, initialize new copy of psi
			eng.run()  # apply U to psi
		Szs.append(psi.expectation_value("Sz"))  # and further measurements...
	return {'beta': betas, 'Sz': Szs}


def computeGibbsState(H, beta_max, dt = 0.05, bc = "finite", order = 2, approx="II"):
	hamiltonian_terms = H.terms
	coefficients = H.coefficients
	n = H.n

	ham_terms_tenpy = [state_simulator.pauliStringToTenpyTerm(n,pauli) for pauli in hamiltonian_terms]
	model_params = dict(L=n, bc_MPS='finite', conserve=None, ham = (ham_terms_tenpy, coefficients))
	M = state_simulator.generalSpinHalfModel(model_params)
	psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
	options = {'trunc_params': {'chi_max': 100, 'svd_min': 1.e-8}}
	beta = 0.
	if order == 1:
		Us = [M.H_MPO.make_U(-dt, approx)]
	elif order == 2:
		Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
	eng = PurificationApplyMPO(psi, Us[0], options)
	Sxs = [psi.expectation_value("Sx")]
	betas = [0.]
	while beta < beta_max:
		beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
		betas.append(beta)
		for U in Us:
			eng.init_env(U)  # reset environment, initialize new copy of psi
			eng.run()  # apply U to psi
		Sxs.append(psi.expectation_value("Sx"))  # and further measurements...
	return {'beta': betas, 'Sx': Sxs, 'psi': psi}


def computeExpectationFromDensityMatrix(pauli_string, rho):
	p = state_simulator.generatePauliMatrix(pauli_string).toarray()
	return np.real(np.trace(rho@p))

def computeGibbsStateED(H, beta):
	n = H.n
	H_terms = H.terms
	H_coeffs = H.coefficients

	H = np.zeros((2**n,2**n), dtype=complex)
	assert len(H_terms) == len(H_coeffs)

	for i in range(len(H_terms)):
		if len(H_terms[i]) != n:
			raise ValueError(f'{H_terms[i]} is not the correct length, expected {n}')
		H += H_coeffs[i]*state_simulator.generatePauliMatrix(H_terms[i]).toarray()

	rho = scipy.linalg.expm(-beta*H)
	rho = rho/np.trace(rho)

	paulis = state_simulator.build2By3CorrelatorList(n)
	return dict([(p,computeExpectationFromDensityMatrix(p, rho)) for p in paulis])

def compareAgainstED(args):
	H= state_simulator.Hamiltonian('./hamiltonians/' + args.hamiltonian_filename + '.txt')
	onebody_operators = utils.buildKLocalPaulis1D(H.n,2, False)
	simulator_params = dict(method = 'MPS', dt = 0.05, bc = "finite", order = 2, approx="II", skip_intermediate_betas = True)

	MPS_expectations_dict, metrics_dict = state_simulator.computeRequiredExpectations(H, onebody_operators, args.beta,
																						simulator_params)
	operators = sorted(list(MPS_expectations_dict.keys()), key = utils.compressPauliToList)

	n = H.n

	print('computing expectations by exact diagonalization')
	t4 = time.time()
	ED_expectations_dict = computeGibbsStateED(H, args.beta)
	t5 = time.time()
	ED_operators = sorted(list(ED_expectations_dict.keys()), key = utils.compressPauliToList)
	assert ED_operators == operators

	### compare m random expectation values
	m=20
	print(f'Comparison of {m} random expectation values:')
	first_line = ' pauli ' + ' '*(n-6) + ' : exp (ED)' + ' : expectation (DMRG)'
	print(first_line)
	print('-'*len(first_line))
	for p in [operators[random.randint(0, len(operators)-1)] for i in range(m)]:
		print(f' {p} : {ED_expectations_dict[p]:+.4f}  '
			+ f' : {MPS_expectations_dict[p]:+.4f}')

	#metrics_dict['rho_computation_time_by_MPO'] = t2-t1
	#metrics_dict['expectations_computation_time_from_MPO'] = t3-t2
	metrics_dict['ED_total_time'] = t5-t4
	metrics_dict['args'] = vars(args)
	metrics_dict['function called'] = 'compareAgainstED'
	metrics_dict['simulator_params'] = simulator_params
	#metrics_dict['tenpy_calls'] = tenpy_calls

	if args.nosave:
		saveEDComparisonResults('.', metrics_dict, MPS_expectations_dict, ED_expectations_dict)
	else:
		save_dir = utils.createSaveDirectory()
		saveEDComparisonResults(save_dir, metrics_dict, MPS_expectations_dict, ED_expectations_dict)

def computeByPurificationOld(H_params, beta):
	metrics_dict = {}
	n = H_params['n']
	k = H_params['k']

	t1 = time.time()
	print('computing purified thermal state by MPO evolution')
	psi = computeGibbsState(H_params,beta)['psi']
	t2 = time.time()
	print('computing thermal state expectations from purified MPS')
	if k == 2 and H_params['locality'] == 'short_range':
		tenpy_calls = [0]
		MPS_expectations_dict = state_simulator.compute2by3correlators(n, psi, tenpy_calls)
	else:
		raise ValueError('havent implemented long-range hamiltonians or k>2')
	t3 = time.time()
	operators = sorted(list(MPS_expectations_dict.keys()), key = utils.compressPauliToList)

	### compare m random expectation values
	m=20
	print(f'Printout of {m} random expectation values:')
	first_line = ' pauli ' + ' '*(n-6) + ' : expectation'
	print(first_line)
	print('-'*len(first_line))
	for p in [operators[random.randint(0, len(operators)-1)] for i in range(m)]:
		print(f' {p} :  {MPS_expectations_dict[p]:+.4f}')

	metrics_dict['rho_computation_time_by_MPO'] = t2-t1
	metrics_dict['expectations_computation_time_from_MPO'] = t3-t2
	metrics_dict['args'] = vars(args)
	metrics_dict['tenpy_calls'] = tenpy_calls

	if args.nosave:
		savePurificationResults('.', metrics_dict, MPS_expectations_dict)
	else:
		save_dir = utils.createSaveDirectory()
		savePurificationResults(save_dir, metrics_dict, MPS_expectations_dict)

def computeByPurification(args):
	H = state_simulator.Hamiltonian(args.hamiltonian_filename)
	onebody_terms = utils.buildKLocalPaulis1D(H.n,2, False)
	expectations_dict, metrics_dict = state_simulator.computeRequiredExpectations(H, onebody_terms, args.beta, simulator_params = {'exp_cache': None})
	operators = sorted(list(expectations_dict.keys()), key = utils.compressPauliToList)

	m=20
	operators_random_selection = [operators[random.randint(0, len(operators)-1)] for i in range(m)]
	r = max([utils.weight(p) for p in operators_random_selection])
	print(f'Printout of {m} random expectation values:')
	first_line = 'pauli' + ' '*(4*r-2) + ': expectation'
	print(first_line)
	print('-'*len(first_line))
	for p in operators_random_selection:
		print(utils.compressPauli(p) + ' '*(4*r - len(utils.compressPauli(p)) + 2) +  f' :  {expectations_dict[p]:+.4f}')

	metrics_dict['args'] = vars(args)
	metrics_dict['function called'] = 'computeByPurification'

	if args.nosave:
		savePurificationResults('.', metrics_dict, expectations_dict)
	else:
		save_dir = utils.createSaveDirectory()
		savePurificationResults(save_dir, metrics_dict, expectations_dict)

def savePurificationResults(save_dir, metrics_dict, expectations_dict):
	os.system(f'cp ./hamiltonians/{args.hamiltonian_filename}.txt {save_dir}/{args.hamiltonian_filename}.txt')
	with open(save_dir + '/metrics.yml', 'w') as f:
		yaml.dump(metrics_dict, f, default_flow_style=False)
	operators = sorted(list(expectations_dict.keys()), key = utils.compressPauliToList)

	r = max([utils.weight(p) for p in operators])

	with open(save_dir + '/results.txt', 'w') as f:
		f.write('results of computeByPurificationOnly in purification_tester.py\n\n')
		line = 'pauli' + ' '*(4*r-2) +':  expectation \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for p in operators:
			f.write(f'{utils.compressPauli(p)}' +
					' '*(4*r - len(utils.compressPauli(p)) + 2) +
					f' :  {expectations_dict[p]:+.10f}\n')

def saveEDComparisonResults(save_dir, metrics_dict, MPS_expectations_dict, ED_expectations_dict):
	os.system(f'cp ./hamiltonians/{args.hamiltonian_filename}.txt {save_dir}/{args.hamiltonian_filename}.txt')
	with open(save_dir + '/metrics.yml', 'w') as f:
		yaml.dump(metrics_dict, f, default_flow_style=False)
	operators = sorted(list(MPS_expectations_dict.keys()), key = utils.compressPauliToList)

	r = max([utils.weight(p) for p in operators])

	diff = [MPS_expectations_dict[p] - ED_expectations_dict[p] for p in operators]
	print(f'mean squared error {np.square(np.linalg.norm(diff))/len(diff)}')

	with open(save_dir + '/results.txt', 'w') as f:
		f.write('results of compareAgainstED in purification_tester.py\n\n')
		f.write(f'mean squared error {np.square(np.linalg.norm(diff))/len(diff)}\n\n')
		line = 'pauli' + ' '*(4*r-2) +':  ED exp         :  MPS exp \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for p in operators:
			f.write(f'{utils.compressPauli(p)}' +
					' '*(4*r - len(utils.compressPauli(p)) + 2) +
					f' :  {ED_expectations_dict[p]:+.10f}  :  {MPS_expectations_dict[p]:+.10f}\n')

def computeGroundstateDMRG(n,hamiltonian_terms, coefficients):
	ham_terms_tenpy = [state_simulator.pauliStringToTenpyTerm(n,pauli) for pauli in hamiltonian_terms]
	model_params = dict(L=n, bc_MPS='finite', conserve=None, ham = (ham_terms_tenpy, coefficients))
	M = state_simulator.generalSpinHalfModel(model_params)
	product_state = ["up"] * n
	psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
	dmrg_params = {
		'mixer': None,  # setting this to True helps to escape local minima
		'max_E_err': 1.e-10,
		'trunc_params': {
			'chi_max': 30,
			'svd_min': 1.e-10
		},
		'combine': True}
	dmrg.run(psi, M, dmrg_params)
	return psi

def groundstateComparison(args):
	beta = args.beta
	H = state_simulator.Hamiltonian('./hamiltonians/' + args.hamiltonian_filename + '.txt')

	n = H.n
	H_paulis = H.terms
	H_coeffs = H.coefficients

	t1 = time.time()
	out_dict = computeGibbsState(H, beta)
	t2 = time.time()
	print(f'n = {n} beta = {beta} time elapsed: {t2-t1} seconds')
	psi_groundstate = computeGroundstateDMRG(H.n,H.terms, H.coefficients)
	gs_sx = psi_groundstate.expectation_value("Sx")

	plt.plot(out_dict['beta'], [sum(x) for x in out_dict['Sx']], label = 'gibbs')
	plt.plot(out_dict['beta'], sum(gs_sx)*np.ones(len(out_dict['beta'])), label = 'groundstate')
	plt.xlabel(r'$\beta$')
	plt.ylabel(r'total $S^x$')
	plt.title(fr'approach to groundstate in $n$ = {n} critical TFI chain')
	plt.savefig(f"./{n}_plot1.pdf", dpi=150)
	plt.show()

	plt.semilogy(out_dict['beta'], [max(np.abs(x-gs_sx)) for x in out_dict['Sx']])
	plt.xlabel(r'$\beta$')
	plt.ylabel(r'max difference in $S^x$')
	plt.title(fr'approach to groundstate in $n$ = {n} critical TFI chain')
	plt.savefig(f"./{n}_plot2.pdf", dpi=150)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('hamiltonian_filename')
	parser.add_argument('beta',type=float)
	parser.add_argument('-ns', '--nosave', action = 'store_true', help = 'skip creating a run directory')
	args = parser.parse_args()

	#H_params = state_simulator.loadHamiltonian(args.hamiltonian_filename)
	'''
	H = state_simulator.Hamiltonian(args.hamiltonian_filename)
	onebody_terms = utils.buildKLocalPaulis1D(H.n,2, False)
	expectations_dict, metrics_dict = state_simulator.computeRequiredExpectations(H, onebody_terms, args.beta, simulator_params = {'exp_cache': None})

	save_dir = utils.createSaveDirectory()
	savePurificationResults(save_dir, metrics_dict, expectations_dict)
	'''

	#computeByPurification(args)
	#groundstateComparison(args)
	compareAgainstED(args)
	
	'''
	n = H_params['n']
	k = H_params['k']
	H_paulis = H_params['terms']
	H_coeffs = H_params['coefficients']
	for i in range(len(H_paulis)):
		print(f'{H_paulis[i]}  {H_coeffs[i]}')
	'''

	#groundstateComparison(H_params,args.beta)
	#computeByPurificationOnly(H_params, args.beta)
	#compareAgainstED(H_params, args.beta)
	

