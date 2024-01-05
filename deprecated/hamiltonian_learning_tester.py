import numpy as np
import matplotlib.pyplot as plt
import scipy
import argparse
import datetime
import os
import projection
import utils
import hamiltonian_learning
import state_simulator
import pickle
import yaml
import functools as ft
import math
import itertools
from tqdm import tqdm

DEFAULT_THRESHOLD = 1e-9
DEFAULT_MU = 0

#C = <a_ia_j> and H = <a_i[H,a_j]>
#computes the onebody hamiltonian "in GNS coordinates", ie. o.n basis in GNS
## TODO: IF THIS MAKES IT INTO THE FINAL PRODUCT, PROBABLY WILL WANT TO REMOVE AUTOMATIC ASSIGNMENT OF THRESHOLD
def oneBodyHamiltonian(C, H_tr_basis, threshold = DEFAULT_THRESHOLD):
	eigvals, eigvecs = scipy.linalg.eigh(C)
	l = len(eigvecs)
	cutoff = 0
	for i in range(l):
		if eigvals[i] > threshold:
			cutoff = i
			break

	D = np.diag(np.reciprocal(np.sqrt(eigvals[cutoff:])))
	E = eigvecs[:,cutoff:]@D
	out = np.conjugate(E.T)@H_tr_basis@E
	return out

## TODO: IF THIS MAKES IT INTO THE FINAL PRODUCT, PROBABLY WILL WANT TO REMOVE AUTOMATIC ASSIGNMENT OF THRESHOLD
def modularOperator(C, threshold = DEFAULT_THRESHOLD):
	eigvals, eigvecs = scipy.linalg.eigh(C)
	l = len(eigvecs)
	cutoff = 0
	for i in range(l):
		if eigvals[i] > threshold:
			cutoff = i
			break
	tprint(f'eigvals discarded: {cutoff}')
	D = np.diag(np.reciprocal(np.sqrt(eigvals[cutoff:])))
	E = eigvecs[:,cutoff:]@D

	out = np.conjugate(E.T)@C.T@E
	return out

def gnsSpectra(save_dir, args, onebody_operators, C,F, H_in, H_learned, expectations_dict, beta, beta_learned, H_manybody_spectrum = None):
	#assert H_in.terms == H_learned.terms
	terms = H_learned.terms
	l = len(terms)
	r = len(onebody_operators)
	n = H_in.n

	in_coeffs_dict = dict(zip(H_in.terms,H_in.coefficients)) #.normalizedCoeffs(expectations_dict)
	in_coeffs = np.zeros(l)
	for i in range(l):
		if terms[i] in in_coeffs_dict:
			in_coeffs[i] = in_coeffs_dict[terms[i]]
	
	learned_coeffs = H_learned.coefficients#.normalizedCoeffs(expectations_dict)

	#H = F.contractRight(H_coeffs).toNumpy()#GNS hamiltonian
	H_orig_tr_basis = (F.toScipySparse()@in_coeffs).reshape((r,r), order = 'F') #learned GNS hamiltonian
	H_learned_tr_basis = (F.toScipySparse()@learned_coeffs).reshape((r,r), order = 'F') #learned GNS hamiltonian

	#print(f'norm of H_gns@H : {np.linalg.norm(H@H_coeffs)}')

	E, logDelta = modularHamiltonian(C)

	H_orig_gns = oneBodyHamiltonian(C, H_orig_tr_basis)

	#print(f'logDelta shape = {logDelta.shape}, H_orig_gns.shape = {H_orig_gns.shape}')
	free_energy_orig_spec = scipy.linalg.eigh(logDelta/beta + H_orig_gns, eigvals_only = True)

	H_learned_gns = oneBodyHamiltonian(C, H_learned_tr_basis)
	free_energy_learned_spec = scipy.linalg.eigh(logDelta/beta_learned + H_learned_gns, eigvals_only = True)

	if H_manybody_spectrum is not None:
		#### normalizing many-body spectrum of initial Hamiltonian. 
		#### This is equivalent to the same normalizations as we required for the learned Hamiltonian
		H_starting_spectrum =  H_starting_spectrum - H_starting_spectrum[0]
		H_starting_spectrum = (2**n)*H_starting_spectrum/sum(H_starting_spectrum)

	##### PLOTTING MODULAR SPECTRUM
	#eigvals = modularSpectrum(onebody_operators, expectations_dict)
	#plt.scatter(np.arange(len(eigvals)), eigvals, s=2)
	#plt.title('modular operator')
	#plt.show()

	if False:#H.n < 11:
		_, spectrum = state_simulator.exactDiagonalization(n, H_paulis, H_coeffs, return_spectrum = True)
	else:
		if args.printing:
			utils.tprint('skipping finding of exact spectrum')

	utils.tprint(f'free energy negativities: original = {min(free_energy_orig_spec)} learned = {min(free_energy_learned_spec)}')

	if args.skip_plotting is False:
		#plt.scatter(np.arange(l), C_eigvals, c='r', s=2, label = 'correlation spectrum')
		#plt.scatter(np.arange(l), H_orig_eigvals, c='b', s=2, label = 'T* H T spectrum') #expect this to be positive (T is GNS map)
		plt.scatter(np.arange(len(free_energy_orig_spec)), free_energy_orig_spec, c='g', s=2, label = 'orig free energy') 
		#plt.scatter(np.arange(l), H_learned_eigvals, c='r', s=2, label = 'reconstructed T* H T spectrum') #expect this to be positive (T is GNS map)
		plt.scatter(np.arange(len(free_energy_learned_spec)), free_energy_learned_spec, c='c', s=2, label = 'reconstructed free energy') 

		if H_manybody_spectrum is not None:
			q = min(2**n,l)
			plt.scatter(np.arange(q), H_starting_spectrum[:q], c='c', s=2, label = 'many-body spectrum of initial Hamiltonian')

		plt.yscale('log')
		plt.title(args.hamiltonian_filename)
		plt.legend()
		plt.savefig(save_dir + "/free_energy_plot.pdf", dpi=150)
		plt.show()


def loadCachedExpectations(exp_cache_filename):
	with open('./caches/' + exp_cache_filename+ '.pkl', 'rb') as f:
		return pickle.load(f)

#DEPRECATED
def computeAllRequiredExpectations(onebody_operators, hamiltonian_terms, state_evaluator):
	print('computing expectations from wavefunction')
	_, threebody_operators = utils.buildTripleProductTensor(onebody_operators, hamiltonian_terms)
	_, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)
	total_operators_list = list(set(onebody_operators + twobody_operators + threebody_operators))
	expectations_dict = dict(zip(total_operators_list,state_evaluator(total_operators_list))) 
	return expectations_dict

def modularSpectrum(onebody_operators, expectations_dict, mult_tensor):
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)
	C = mult_tensor.contractRight([expectations_dict[p] for p in twobody_operators]).toNumpy()
	Delta = modularOperator(C)
	eigvals = scipy.linalg.eigh(Delta, eigvals_only = True)
	return eigvals

## TODO: IF THIS MAKES IT INTO THE FINAL PRODUCT, PROBABLY WILL WANT TO REMOVE AUTOMATIC ASSIGNMENT OF THRESHOLD
def modularHamiltonian(C, threshold = DEFAULT_THRESHOLD):
	eigvals, eigvecs = scipy.linalg.eigh(C)
	l = len(eigvecs)
	cutoff = 0
	for i in range(l):
		if eigvals[i] > threshold:
			cutoff = i
			break

	utils.tprint(f'eigvals discarded: {cutoff}')
	D = np.diag(np.reciprocal(np.sqrt(eigvals[cutoff:])))
	E = eigvecs[:,cutoff:]@D

	Delta = np.conjugate(E.T)@C.T@E
	logDelta = scipy.linalg.logm(Delta)
	return E, logDelta

#save_dir, metrics_and_params, H_in,H_learned)
def saveLearningResults(metrics_and_params, H_in, T_in, H_learned, T_learned, expectations_dict, args):
	if args.nosave:
		save_dir = '.'
	else:
		save_dir = utils.createSaveDirectory()

	hamiltonian_filename = args.hamiltonian_filename
	n = H_in.n

	#os.system(f'cp ./hamiltonians/{hamiltonian_filename}.yml {save_dir}/{hamiltonian_filename}.yml')
	if args.disorder > 0:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}_disordered.yml')
	else:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}.yml')

	with open(save_dir + '/metrics_and_params.yml', 'w') as f:
		yaml.dump(metrics_and_params, f, default_flow_style=False)

	terms = H_learned.terms

	r = max([utils.weight(p) for p in terms])
	l = len(terms)

	H_in.addTerms(['I'*n], [1])
	H_in.normalizeCoeffs(expectations_dict)
	H_learned.normalizeCoeffs(expectations_dict)

	in_coeffs_dict = dict(zip(H_in.terms,H_in.coefficients))
	learned_coeffs_dict = dict(zip(H_learned.terms,H_learned.coefficients))

	learned_coeffs = np.asarray([learned_coeffs_dict[p] for p in terms])
	in_coeffs = np.zeros(len(terms))
	for i in range(len(terms)):
		p = terms[i]
		if p in in_coeffs_dict:
			in_coeffs[i] = in_coeffs_dict[p]

	utils.tprint(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}')
	if args.printing:
		utils.tprint(f'norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = args.obj_ord)}')
		utils.tprint(f'norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = args.obj_ord)}')
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

	with open(save_dir + f'/{args.routine_name}_results.txt', 'w') as f:
		f.write(f'results of hamiltonian_learning_tester.py subroutine {args.routine_name}\n\n')
		f.write(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}\n\n')
		f.write(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}\n\n')
		f.write(f'norm of original Hamiltonian: {np.linalg.norm(in_coeffs[1:], ord = args.obj_ord)}\n')
		f.write(f'norm of learned Hamiltonian: {np.linalg.norm(learned_coeffs[1:], ord = args.obj_ord)}\n\n')
		line = 'term' + ' '*(4*r-1) +':  orig coeff     :  learned coeff \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for i in range(l):
			f.write(f'{utils.compressPauli(terms[i])}' +
					' '*(4*r - len(utils.compressPauli(terms[i])) + 2) +
					f' :  {in_coeffs[i]:+.10f}  :  {learned_coeffs[i]:+.10f}\n')

	return save_dir

#save_dir, metrics_and_params, H_in,H_learned)
def saveModularHamiltonianLearningResults(save_dir, metrics_and_params, H_in, T_in, H_learned, T_learned, expectations_dict, args):
	hamiltonian_filename = args.hamiltonian_filename
	n = H_in.n

	#os.system(f'cp ./hamiltonians/{hamiltonian_filename}.yml {save_dir}/{hamiltonian_filename}.yml')
	if args.disorder >0:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}_disordered.yml')
	else:
		H_in.saveToYAML(f'{save_dir}/{n}_{hamiltonian_filename}.yml')

	with open(save_dir + '/metrics_and_params.yml', 'w') as f:
		yaml.dump(metrics_and_params, f, default_flow_style=False)

	terms = H_learned.terms

	r = max([utils.weight(p) for p in terms])
	l = len(terms)

	in_coeffs_dict = dict(zip(H_in.terms,H_in.coefficients))
	learned_coeffs_dict = dict(zip(H_learned.terms,H_learned.coefficients))

	#print(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}')
	if args.printing:
		print()
		line = 'term' + ' '*(4*r-1) +':  orig coeff :  mod ham coeff'
		print(line)
		print('-'*len(line))
		for p in terms:
			if p in in_coeffs_dict:
				in_coeff = in_coeffs_dict[p]
			else:
				in_coeff = 0
			learned_coeff = learned_coeffs_dict[p]

			if max(np.abs(in_coeff),np.abs(learned_coeff)) > 1e-10:
				print(f'{utils.compressPauli(p)}' +
					' '*(4*r - len(utils.compressPauli(p)) + 2) +
					f' :  {in_coeff:+.6f}  :  {learned_coeff:+.6f}')
		print()

	with open(save_dir + '/testModularHamiltonianLearning_results.txt', 'w') as f:
		f.write('results of hamiltonian_learning_tester.py subroutine testThermalStateLearning\n\n')
		if T_learned == None:
			f.write(f'original T = {T_in:.10e}   learned T = {T_learned}\n\n')
		else:
			f.write(f'original T = {T_in:.10e}   learned T = {T_learned:.10e}\n\n')
		#f.write(f'average squared reconstruction error is {np.square(np.linalg.norm(in_coeffs - learned_coeffs))/l}\n\n')
		line = 'term' + ' '*(4*r-1) +':  orig coeff     :  mod ham coeff \n'
		f.write(line)
		f.write('-'*(len(line)-1)+'\n')
		for p in terms:
			if p in in_coeffs_dict:
				in_coeff = in_coeffs_dict[p]
			else:
				in_coeff = 0
			learned_coeff = learned_coeffs_dict[p]

			f.write(f'{utils.compressPauli(p)}' +
					' '*(4*r - len(utils.compressPauli(p)) + 2) +
					f' :  {in_coeff:+.10f}  :  {learned_coeff:+.10f}\n')


	if args.skip_plotting is False:
		if args.hamiltonian_filename == 'tf_ising_ferro':
			ZZ_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Z {k} Z {k+1}',n)] for k in range(n-1)])
			X_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'1 X {k}',n)] for k in range(n)])
			normalization = -0.5/X_learned_coeffs[-1]
			ZZ_learned_coeffs = normalization*ZZ_learned_coeffs
			X_learned_coeffs = normalization*X_learned_coeffs

			h = -args.g
			k = min(1/abs(h),abs(h))
			if np.abs(h) < 1:
				##ordered phase
				X_theoretical_coeffs = [-k*(i + 0.5) for i in range(n)]
				ZZ_theoretical_coeffs = [-(i + 1) for i in range(n-1)]
				X_theoretical_coeffs.reverse()
				ZZ_theoretical_coeffs.reverse()
			else:
				##disordered phase
				X_theoretical_coeffs = [-(i + 0.5) for i in range(n)]
				ZZ_theoretical_coeffs = [-k*(i + 1) for i in range(n-1)]
				X_theoretical_coeffs.reverse()
				ZZ_theoretical_coeffs.reverse()

			plt.scatter(np.arange(n), X_learned_coeffs, s=2, label = 'X coefficient')
			plt.scatter(np.arange(n-1), ZZ_learned_coeffs, s=2, label = 'ZZ coefficient')
			plt.scatter(np.arange(n), X_theoretical_coeffs, s=2, label = 'X coefficient (inf-volume theoretical)')
			plt.scatter(np.arange(n-1), ZZ_theoretical_coeffs, s=2, label = 'ZZ coefficient (inf-volume theoretical)')
			plt.xlabel('site')
			plt.title(f'n= {args.n} g = {args.g} tfi modular hamiltonian on half-line')
			plt.legend()
			plt.savefig(save_dir + "/reconstructed_hamiltonian.pdf", dpi=150)
			plt.show()

		elif args.hamiltonian_filename == 'xxz_Jneg':
			XX_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 X {k} X {k+1}',n)] for k in range(n-1)])
			YY_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Y {k} Y {k+1}',n)] for k in range(n-1)])
			ZZ_learned_coeffs = np.asarray([learned_coeffs_dict[utils.decompressPauli(f'2 Z {k} Z {k+1}',n)] for k in range(n-1)])
			if XX_learned_coeffs[-1] > 1e-3:
				normalization = XX_learned_coeffs[-1]
			else:
				print('skipping normalizing because learned XX coefficient on the right is too small')
				normalization = 1
			XX_learned_coeffs = XX_learned_coeffs/normalization
			YY_learned_coeffs = YY_learned_coeffs/normalization
			ZZ_learned_coeffs = ZZ_learned_coeffs/normalization

			XX_theoretical_coeffs = [n-1-i for i in range(n-1)]
			YY_theoretical_coeffs = [n-1-i for i in range(n-1)]
			ZZ_theoretical_coeffs = [args.g*(n-1-i) for i in range(n-1)]

			colors = ['r','g','b']

			plt.plot(np.arange(n-1), XX_theoretical_coeffs, c = 'r', label = 'XX coefficient (inf-volume theoretical)')
			plt.plot(np.arange(n-1), YY_theoretical_coeffs, c = 'g', label = 'YY coefficient (inf-volume theoretical)')
			plt.plot(np.arange(n-1), ZZ_theoretical_coeffs, c = 'b', label = 'ZZ coefficient (inf-volume theoretical)')
			plt.scatter(np.arange(n-1), XX_learned_coeffs, s=2, c = 'r', label = 'XX coefficient')
			plt.scatter(np.arange(n-1), YY_learned_coeffs, s=2, c = 'g', label = 'YY coefficient')
			plt.scatter(np.arange(n-1), ZZ_learned_coeffs, s=2, c = 'b', label = 'ZZ coefficient')
			plt.xlabel('site')
			plt.title(f'n= {args.n} g = {args.g} xxz modular hamiltonian on half-line')
			plt.legend()
			plt.savefig(save_dir + "/reconstructed_hamiltonian.pdf", dpi=150)
			plt.show()

def GNSFreeEnergySpectrum(C,C_eigvals,C_eigvecs, cutoff, hamiltonian_terms ,H,F, lam = 0, beta = np.inf):
	l = len(C_eigvals)
	H_coeffs_dict = dict(zip(H.terms,H.coefficients))

	#print(H_coeffs_dict)

	H_coeffs_list = [H_coeffs_dict[p] if p in H_coeffs_dict else 0 for p in hamiltonian_terms]

	#print(H_coeffs_list)
	
	D = np.diag(np.reciprocal(np.sqrt(C_eigvals[cutoff:])+lam))
	E = C_eigvecs[:,cutoff:]@D

	#print(f'C_eigvals = {C_eigvals}')
	#print(f'C_eigvecs[4] = {C_eigvecs[4]}')

	Delta = np.conjugate(E.T)@(C+lam).T@E
	logDelta = scipy.linalg.logm(Delta)

	logDeltaEigs = scipy.linalg.eigh(logDelta, eigvals_only = True)

	#print(f'logDeltaEigs = {logDeltaEigs}')
	#print(f'lam = {lam}')

	#print(f'lambda = {lam},  logDelta max eigenvector {logDeltaEigs[-1]:.4e}  logDelta min eigenvector {logDeltaEigs[0]:.4e}')
	#print(f'beta = {args.beta}')
	if beta == np.inf:
		free_energy = np.conjugate(E.T)@(F@H_coeffs_list).reshape((l,l), order = 'F')@E
	else:
		free_energy = logDelta + beta*np.conjugate(E.T)@(F@H_coeffs_list).reshape((l,l), order = 'F')@E

	free_energy_spectrum = scipy.linalg.eigh(free_energy, eigvals_only=True)
	
	#print(f'free_energy_spectrum = {free_energy_spectrum}')
	return free_energy_spectrum

def GNSFreeEnergySpectrum2(C,C_eigvals,C_eigvecs, cutoff, H_gns_tr_basis, lam = 0, beta = np.inf):
	l = len(C_eigvals)
	
	D = np.diag(np.reciprocal(np.sqrt(C_eigvals[cutoff:])+lam))
	E = C_eigvecs[:,cutoff:]@D

	Delta = np.conjugate(E.T)@(C+lam).T@E
	logDelta = scipy.linalg.logm(Delta)

	logDeltaEigs = scipy.linalg.eigh(logDelta, eigvals_only = True)

	if beta == np.inf:
		free_energy = np.conjugate(E.T)@(H_gns_tr_basis).reshape((l,l), order = 'F')@E
	else:
		free_energy = logDelta + beta*np.conjugate(E.T)@(H_gns_tr_basis).reshape((l,l), order = 'F')@E

	free_energy_spectrum = scipy.linalg.eigh(free_energy, eigvals_only=True)
	
	return free_energy_spectrum


pauli_generators = {'X': np.array([[0,1],[1,0]], dtype = complex),
					'Y': np.array([[0,-1.j],[0.+1.j,0]]),
					'Z': np.array([[1.,0.],[0,-1.]], dtype=complex),
					'I': np.identity(2, dtype=complex)}

def pauliMatrix(pauli_string):
	pauli_list = [(pauli_generators[c]) for c in pauli_string]
	return ft.reduce(np.kron, pauli_list)

def computeExpectation(pauli_string, rho):
	pauli_matrix = pauliMatrix(pauli_string)
	return np.real(np.trace(rho@pauli_matrix))

def sqrt(m):
	eigvals, eigvecs = scipy.linalg.eigh(m)
	D = np.diag(np.sqrt(eigvals))
	return eigvecs@D@np.conjugate(eigvecs.T)

def invSqrt(m):
	eigvals, eigvecs = scipy.linalg.eigh(m)
	D = np.diag(np.reciprocal(np.sqrt(eigvals)))
	return eigvecs@D@np.conjugate(eigvecs.T)

def computeThermalStateED(H,beta):
	load = True
	n = H.n
	### ensure all hamiltonian terms are there
	#print('Hamiltonian:')
	#for i in range(len(H.terms)):
	#	print(f'{H.terms[i]}  :  {H.coefficients[i]}')

	print('building many-body hamiltonian matrix')
	### build many-body hamiltonian matrix
	H_mb = np.zeros((2**n,2**n), dtype = complex)
	for i in range(len(H.terms)):
		if len(H.terms[i]) != n:
			raise ValueError(f'{H.terms[i]} is not the correct length, expected {n}')
		H_mb += H.coefficients[i]*pauliMatrix(H.terms[i])

	### check H_mb is selfadjoint
	H_mb_dag = np.conjugate(H_mb.T)
	assert np.array_equal(H_mb, H_mb_dag)

	#print('printing pauli generators')
	#for pauli_string in pauli_generators:
	#	p = pauliMatrix(pauli_string)
	#	print(p)
	#	assert np.array_equal(np.conjugate(p.T), p)

	print('computing state')
	### compute state
	if beta == np.inf:
		_, eigvecs = scipy.linalg.eigh(H_mb)
		psi = eigvecs[:,0]
		rho = np.conjugate(psi.reshape((2**n,1)))@psi.reshape((1,2**n))
	else:
		rho = scipy.linalg.expm(-beta*H_mb)
		rho = rho/np.trace(rho)

	print(f'rho.shape = {rho.shape}')
	return rho

def computeExpectationsED(rho, operators):
	out = []
	for p in tqdm(operators):
		out.append(computeExpectation(p,rho))
	return out

def computeOneParticleFreeEnergyED(onebody_operators, H, beta):
	rho = computeThermalStateED(H,beta)

	### building C matrix
	#C = np.tile(np.nan, (l, l))
	C = 500*np.ones((l,l), dtype = complex)

	#e = computeExpectation('ZZIII', rho)
	#print(f'expectation of ZZIII : {e:.2f}')

	for i in range(l):
		for j in range(l):
			pauli_1 = onebody_operators[i]
			pauli_2 = onebody_operators[j]
			W,w = utils.multiplyPaulis(pauli_1,pauli_2)
			C[i,j] = w*computeExpectation(W, rho)

			### checking that multiplication does what it's supposed to
			Z,z = utils.multiplyPaulis(pauli_2,pauli_1)
			if Z != W or np.conjugate(z) != w:
				print(f'a = {pauli_1} b = {pauli_2}  ab = {w}{W}  ba = {z}{Z}')

	C_dag = np.conjugate(C.T)

	#eqs = np.equal(C, 500*np.ones((l,l), dtype = complex))
	eqs = np.abs(C-C_dag)<1e-8

	for i,j in itertools.product(range(l), range(l)):
		if not eqs[i,j]:
			#print(f'entry ({i},{j}) was not updated')
			a_i = onebody_operators[i]
			a_j = onebody_operators[j]
			print(f'a_i = {a_i}, a_j = {a_j} C[{i},{j}] = {C[i,j]:.2f}, C^dag[{i},{j}] = {C_dag[i,j]:.2f} C[{j},{i}]* = {np.conjugate(C[j,i]):.2f}')

	print(f'norm of C^dagger - C : {np.linalg.norm(np.conjugate(C.T) - C)}')

	### building H matrix
	#H_onebody = np.tile(np.nan, (l, l))
	H_onebody = np.zeros((l,l), dtype = complex)

	if load:
		with open('./hamiltonian_tmp.pkl', 'rb') as f:
			H_onebody = pickle.load(f)
	else:
		for i in range(l):
			for j in range(l):
				if n>2:
					if (l*i + j)%(l**2//20) == 1:
						print(f'computing Hamiltonian {math.ceil(100*((l*i+j)/l**2))}% done') 
				p_sum = 0
				pauli_1 = onebody_operators[i]
				pauli_2 = onebody_operators[j]
				for k in range(len(H.terms)):
					p = H.terms[k]
					A,a = utils.multiplyPaulis(pauli_1,p)
					B,b = utils.multiplyPaulis(A,pauli_2)

					if (i,j) == (0,30) and np.abs(a*b*computeExpectation(B, rho)*H.coefficients[k]) > 1e-10:
						print(f'p = {p}. a_i = {pauli_1} a_j = {pauli_2}. adding {a*b*computeExpectation(B, rho)*H.coefficients[k]} to H[{i}][{j}]')

					p_sum += a*b*computeExpectation(B, rho)*H.coefficients[k]

					A,a = utils.multiplyPaulis(pauli_1,pauli_2)
					B,b = utils.multiplyPaulis(A,p)

					p_sum -= a*b*computeExpectation(B, rho)*H.coefficients[k]

					if (i,j) == (0,30) and np.abs(a*b*computeExpectation(B, rho)*H.coefficients[k]) > 1e-10:
						print(f'p = {p}. a_i = {pauli_1} a_j = {pauli_2}. adding {-a*b*computeExpectation(B, rho)*H.coefficients[k]} to H[{i}][{j}]')

				H_onebody[i,j] = p_sum
	print()
	print()

	with open('./hamiltonian_tmp.pkl', 'wb') as f:
		pickle.dump(H_onebody,f)

	### checking H is selfadjoint

	H_onebody_dag = np.conjugate(H_onebody.T)

	#eqs = np.equal(C, 500*np.ones((l,l), dtype = complex))
	eqs = np.abs(H_onebody-H_onebody_dag)<1e-8

	#paulis = onebody_operators
	#for p in paulis:
	#	print(f'expectation of {p}: {computeExpectation(p,rho)}')

	#print(f'H_01: {H_onebody[0][1]}')

	for i,j in itertools.product(range(l), range(l)):
		if not eqs[i,j]:
			#print(f'entry ({i},{j}) was not updated')
			a_i = onebody_operators[i]
			a_j = onebody_operators[j]
			print(f'a_i = {a_i}, a_j = {a_j} H[{i},{j}] = {H_onebody[i,j]:.2f}, H^dag[{i},{j}] = {H_onebody_dag[i,j]:.2f}')# H[{j},{i}]* = {np.conjugate(H_onebody[j,i]):.2f}')

	print(f'norm of H^dagger - H : {np.linalg.norm(H_onebody_dag - H_onebody)}')

	### computing free energy
	E = invSqrt(C)
	E_inv = sqrt(C)

	assert np.all(np.abs(E_inv@E - np.eye(l)) < 1e-8)

	Delta = E@C.T@E
	logDelta = scipy.linalg.logm(Delta)

	G_gns = logDelta + beta*E@H_onebody@E

	#G_tr = C.T + beta*H_onebody
	'''
	for tau in np.arange(0,3, 0.2):
		G_gns = logDelta + tau*E@H_onebody@E
		G_gns_spec = scipy.linalg.eigh(G_gns,eigvals_only = True)
		plt.scatter(range(len(G_gns_spec)), G_gns_spec, s=2, label = fr'$tau$ = {tau}')

	plt.legend()
	plt.title(r'one-particle free energy by ED for different values of $tau$')
	plt.show()
	'''
	_,twobody_operators = utils.buildMultiplicationTensor(onebody_operators)
	triple_product_tensor, threebody_operators = utils.buildTripleProductTensor(onebody_operators, onebody_operators)
	triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	print('computing ED expectations')
	twobody_expectations = [computeExpectation(p,rho) for p in twobody_operators]
	threebody_expectations = [computeExpectation(p,rho) for p in threebody_operators]
	F_prime = triple_product_tensor.contractRight(threebody_expectations)
	F_prime.vectorize(axes = [0,1])


	return C,F_prime, H_onebody, G_gns, rho

def getMPSQuantities(args):
	beta = args.beta
	H, onebody_operators, hamiltonian_terms, simulator_params, learning_params = parseCommandLineInput(args)

	expectations_dict, metrics = state_simulator.requiredExpectations(H, onebody_operators, hamiltonian_terms, args.beta, simulator_params)

	metrics_dict = {}
	metrics_and_params = dict(metrics_dict = metrics_dict, args_at_runtime = vars(args), simulator_params = simulator_params)

	############## extra stuff for debugging
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)
	triple_product_tensor, threebody_operators = utils.buildTripleProductTensor(onebody_operators, onebody_operators, args.printing)
	triple_product_tensor.transpose([0,2,1,3])#want the Hamiltonian index to be second-last
	twobody_expectations = [expectations_dict[p] for p in twobody_operators]
	threebody_expectations = [expectations_dict[p] for p in threebody_operators]
	F_prime = triple_product_tensor.contractRight(threebody_expectations)
	F_prime.vectorize(axes = [0,1])
	#F = triple_product_tensor.contractRight(threebody_expectations).toNumpy()#F_ijk = <a_i[a_k,a_j]> (note the order of indices)
	l = len(onebody_operators)
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()#C_ij = <a_ia_j

	'''
	_,spectrum = state_simulator.exactDiagonalization(H.n,H.terms,H.coefficients, return_spectrum = True)

	spectrum = spectrum-spectrum[0]

	plt.scatter(range(len(spectrum)), spectrum,s=2)
	plt.title("manybody spectrum")
	plt.show()
	plt.cla()

	x = np.exp(-args.beta*spectrum)
	x = x/np.sum(x)

	print(f'beta = {args.beta}')

	plt.scatter(range(len(x)),x,s=2)
	plt.title("manybody gibbs weights")
	plt.show()

	thresholds = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1, 2]

	for thresh in thresholds:
		Delta = modularOperator(C, thresh)
		delta_eigs = scipy.linalg.eigh(Delta, eigvals_only = True)
		print(f'threshold = {thresh}, lowest 10 eigvalues of modular operator are {delta_eigs[:10]}')
		plt.scatter(range(len(delta_eigs)), delta_eigs, s=2, label = f'threshold = {thresh}')

	plt.yscale('log')
	plt.legend()
	plt.title('modular operator at different thresholds')
	plt.plot()
	plt.show()
	plt.cla()

	for thresh in thresholds:
		H_tr_basis = (F_prime.toScipySparse()@H.coefficients).reshape((l,l), order = 'F')
		H_onebody = oneBodyHamiltonian(C,H_tr_basis, thresh)
		delta_eigs = scipy.linalg.eigh(H_onebody, eigvals_only = True)
		print(f'threshold = {thresh}, lowest 10 eigvalues of modular operator are {delta_eigs[:10]}')
		plt.scatter(range(len(delta_eigs)), delta_eigs, s=2, label = f'threshold = {thresh}')

	plt.legend()
	plt.title('GNS Hamiltonian at different thresholds')
	plt.plot()
	plt.show()
	plt.cla()

	lambdas = [0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
	'''

	eigvals, eigvecs = scipy.linalg.eigh(C)
	l = len(eigvals)
	cutoff = 0
	for i in range(l):
		if eigvals[i] > learning_params['threshold']:
			cutoff = i
			break
	print(f'cutoff = {cutoff}')

	return C,eigvals,eigvecs,cutoff, onebody_operators, H, F_prime, learning_params

def compareQuantities(args):
	H, onebody_operators, hamiltonian_terms, simulator_params, learning_params = parseCommandLineInput(args)
	metrics_dict = {}

	triple_product_tensor, threebody_operators = utils.buildTripleProductTensor(onebody_operators, H.terms, args.printing)

	C_ED,F_prime_ED, H_tr_basis_ED, G_gns_ED, rho = computeOneParticleFreeEnergyED(onebody_operators, H, args.beta)

	expectations_dict, metrics = state_simulator.requiredExpectations(H, onebody_operators, args.beta, simulator_params)

	#threebody_operators = list(expectations_dict.keys())
	#threebody_expectations_ED = np.asarray([computeExpectation(p,rho) for p in threebody_operators])
	#threebody_expectations_MPS = np.asarray([expectations_dict[p] for p in threebody_operators])

	eigvals_ED, eigvecs_ED = scipy.linalg.eigh(C_ED)
	l = len(eigvals_ED)
	cutoff_ED = 0
	threshold = learning_params['threshold']
	for i in range(l):
		if eigvals_ED[i] > threshold:
			cutoff_ED = i
			break
	print(f'cutoff_ED = {cutoff_ED}')

	C_MPS,eigvals_MPS,eigvecs_MPS,cutoff_MPS, onebody_terms, H, F_prime_MPS, learning_params = getMPSQuantities(args)
	spec_MPS = GNSFreeEnergySpectrum(C_MPS,eigvals_MPS,eigvecs_MPS,cutoff_MPS, hamiltonian_terms, H, F_prime_MPS.toScipySparse(), beta=args.beta)
	#spec_ED = GNSFreeEnergySpectrum2(C_ED,eigvals_ED,eigvecs_ED, cutoff_ED, H_tr_basis_ED, beta = args.beta)
	
	#plt.scatter(range(len(spec_ED)), spec_ED, s=2, label = 'ED')
	plt.scatter(range(len(spec_MPS)), spec_MPS, s=2, label = 'MPS')
	plt.title(f'GNS free energy spectra at threshold {threshold}')
	plt.legend()
	plt.show()

	'''
	errors = threebody_expectations_MPS - threebody_expectations_ED

	plt.scatter(threebody_expectations_ED, errors, s=1)
	plt.title('mps errors')
	plt.show()

	print(f'mean squared error : {np.square(np.linalg.norm(errors))/len(threebody_operators)}')

	for i in range(len(threebody_operators)):
		e_ED = threebody_expectations_ED[i]
		e_MPS = threebody_expectations_MPS[i]
		if np.abs(e_ED-e_MPS)>1e-3:
			assert np.imag(e_ED) < 1e-10
			print(f'p = {threebody_operators[i]}  ED exp: {np.real(e_ED):+.2e}    MPS exp: {e_MPS:+.2e}')
	'''
	return C_MPS,eigvals_MPS,eigvecs_MPS,cutoff_MPS, onebody_terms, H, F_prime_MPS

possible_coupling_constants = ['g','Delta']

def generateHamiltonianTerms(n, k, locality, periodic, l):
	if locality == 'short_range':
		return utils.buildKLocalPaulis1D(n,k,periodic)
	elif locality == 'long_range':
		return utils.buildKLocalCorrelators1D(n,k,periodic,l)
	else:
		raise ValueError(f'generateHamiltonianTerms got unrecognized argument for locality: {locality}')

def parseCommandLineInput(args):
	onebody_operators = utils.buildKLocalPaulis1D(args.n, args.k, args.periodic)

	hamiltonian_terms = generateHamiltonianTerms(args.n, args.H_k, args.H_locality, args.periodic, args.H_range)

	coupling_constants_dict = dict([(g, vars(args)[g]) for g in possible_coupling_constants if vars(args)[g] is not None])
	H = state_simulator.Hamiltonian(args.n, f'./hamiltonians/{args.hamiltonian_filename}.yml', coupling_constants_dict)

	simulator_params = dict(method = 'MPS', 
							dt = 0.001, 
							bc = "finite", 
							order = 2, 
							approx="II", 
							skip_intermediate_betas = True, 
							skip_checks = args.nochecks,
							overwrite_cache = args.overwrite_cache,
							printing = args.printing)

	mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8}

	learning_params = dict(mu = args.mu, 
						threshold = args.threshold, 
						obj_ord = args.obj_ord, 
						mosek_params = mosek_params,
						printing = args.printing)

	if args.nocache:
		simulator_params['exp_cache'] = None
	elif args.disorder > 0:
		print('skipping loading of expectations because disorder is nonzero')
		simulator_params['exp_cache'] = None
	else:
		if len(coupling_constants_dict) == 0:
			simulator_params['exp_cache'] = f'./caches/{args.hamiltonian_filename}_exp_cache.hdf5'
		elif len(coupling_constants_dict) == 1:
			g = coupling_constants_dict['g']
			n = H.n
			simulator_params['exp_cache'] = f'./caches/{n}_{args.hamiltonian_filename}_g={g}_exp_cache.hdf5'
		else:
			raise ValueError('need find a better way of naming caches')

	if args.disorder > 0:
		print(f'adding disorder to Hamiltonian (magnitude = {args.disorder})')
		H.addDisorder(args.disorder)

	return H, onebody_operators, hamiltonian_terms, simulator_params, learning_params

def plotModularSpectrum(onebody_operators, expectations_dict, threshold):
	mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)
	twobody_expectations = [expectations_dict[p] for p in twobody_operators]
	C = mult_tensor.contractRight(twobody_expectations).toNumpy()
	eigvals, eigvecs = scipy.linalg.eigh(C)
	l = len(eigvals)
	cutoff = 0
	for i in range(l):
		if eigvals[i] > threshold:
			cutoff = i
			break
	utils.tprint(f'dropped {cutoff} lowest eigenvalues from covariance matrix C')
	plt.scatter(np.arange(l-cutoff), eigvals[cutoff:], s=2)
	plt.title('modular spectrum')
	plt.show()

def testModularHamiltonianLearning(args):
	H, onebody_operators, hamiltonian_terms, simulator_params, learning_params = parseCommandLineInput(args)
	n = H.n
	#k = n//2
	region = list(range(n//2))
	m = len(region)

	expectations_dict, metrics = state_simulator.requiredExpectations(H, onebody_operators, hamiltonian_terms, np.inf, simulator_params)

	metrics_dict = {}
	metrics_and_params = dict(metrics_dict = metrics_dict, args_at_runtime = vars(args), simulator_params = simulator_params, region = region)

	if args.skip_plotting is False:
		plotModularSpectrum(onebody_operators, expectations_dict, learning_params['threshold'])

	#embed = lambda p : p+'I'*(n-n//2)
	onebody_operators_restricted = utils.restrictOperators(n, onebody_operators, region)
	hamiltonian_terms_restricted = utils.restrictOperators(n, hamiltonian_terms, region)
	#evaluator = lambda p: expectations_dict[p+'I'*(n-n//2)]
	evaluator = lambda p: expectations_dict[utils.embedPauli(n, p,region)]
	
	metrics_and_params['metrics_dict']['expectations_stats'] = metrics
	learned_coeffs, T_learned, C, F, solver_stats = hamiltonian_learning.learnHamiltonianFromThermalState(len(region), 
		onebody_operators_restricted, hamiltonian_terms_restricted, evaluator, learning_params)
	metrics_and_params['metrics_dict']['solver_stats'] = solver_stats

	if learned_coeffs is None:
		H_learned = state_simulator.Hamiltonian(m, hamiltonian_terms_restricted, np.tile(np.nan, len(hamiltonian_terms_restricted)))
	else:
		H_learned = state_simulator.Hamiltonian(m, hamiltonian_terms_restricted, learned_coeffs)

	if H_learned.n < 12:
		### generate gibbs state and compare
		print(f'computing thermal state by ED with beta = {1/T_learned}')
		rho = computeThermalStateED(H_learned,1/T_learned)
		
		print('computing expectations by ED')
		learned_ham_expectations = computeExpectationsED(rho, onebody_operators_restricted)
		true_expectations = [evaluator(p) for p in onebody_operators_restricted]
		for i in range(len(onebody_operators_restricted)):
			p = onebody_operators_restricted[i]
			print(f'pauli: {p} true exp: {true_expectations[i]:+.5f}  learned exp: {learned_ham_expectations[i]:+.5f}')

	save_dir = utils.createSaveDirectory()
	saveModularHamiltonianLearningResults(save_dir, metrics_and_params, H.restricted(region), 0, H_learned, T_learned, expectations_dict, args)

def testThermalStateLearning(args):
	beta = args.beta
	H, onebody_operators, hamiltonian_terms, simulator_params, learning_params = parseCommandLineInput(args)
	n = H.n
	threshold = learning_params['threshold']

	expectations_dict, metrics = state_simulator.requiredExpectations(H, onebody_operators, hamiltonian_terms, args.beta, simulator_params)

	metrics_dict = {}
	metrics_and_params = dict(metrics_dict = metrics_dict, args_at_runtime = vars(args), simulator_params = simulator_params)

	expectations_evaluator = lambda x: expectations_dict[x]
	
	learned_coeffs, T_learned, C,F, solver_stats = hamiltonian_learning.learnHamiltonianFromThermalState(H.n, 
		onebody_operators, hamiltonian_terms, expectations_evaluator, learning_params)

	metrics_and_params['metrics_dict']['expectations_stats'] = metrics
	metrics_and_params['metrics_dict']['solver_stats'] = solver_stats

	if learned_coeffs is None:
		H_learned = state_simulator.Hamiltonian(n, hamiltonian_terms, np.tile(np.nan, len(hamiltonian_terms)))
	else:
		H_learned = state_simulator.Hamiltonian(n, hamiltonian_terms, learned_coeffs)

	if False:#H.n < 11:
		_, spectrum = state_simulator.exactDiagonalization(n, H_paulis, H_coeffs, return_spectrum = True)
	else:
		print('skipping finding of exact spectrum')

	save_dir = saveLearningResults(metrics_and_params, H, 1/(H.getNormalization()*args.beta), H_learned, T_learned, expectations_dict, args)

	gnsSpectra(save_dir,args, onebody_operators, C,F,H ,H_learned, expectations_dict, args.beta*H.getNormalization(), 1/T_learned)

def testGroundstateLearning(args):
	H, onebody_operators, hamiltonian_terms, simulator_params, learning_params = parseCommandLineInput(args)
	n = H.n
	threshold = learning_params['threshold']

	### getting the dict of expectations values for the given satate
	expectations_dict, expectations_metrics = state_simulator.requiredExpectations(H, onebody_operators, hamiltonian_terms, np.inf, simulator_params)

	### learning Hamiltonian coefficients
	expectations_evaluator = lambda x: expectations_dict[x]
	learned_coeffs, C,F, solver_stats = hamiltonian_learning.learnHamiltonianFromGroundstate(H.n, onebody_operators, hamiltonian_terms, expectations_evaluator, learning_params)

	### constructing Hamiltonian instance from learned coefficients
	if learned_coeffs is None:
		H_learned = state_simulator.Hamiltonian(n, hamiltonian_terms, np.tile(np.nan, len(hamiltonian_terms)))
	else:
		H_learned = state_simulator.Hamiltonian(n, hamiltonian_terms, learned_coeffs)

	### packaging some metrics
	metrics_dict = dict(solver_stats = solver_stats, expectations_metrics = expectations_metrics)
	metrics_and_params = dict(metrics_dict = metrics_dict, args_at_runtime = vars(args), simulator_params = simulator_params)

	### saving results to a new directory in ./runs/
	save_dir = saveLearningResults(metrics_and_params, H, 0, H_learned, 0, expectations_dict, args)

	### plotting
	gnsSpectra(save_dir, args, onebody_operators, C,F,H ,H_learned, expectations_dict, np.inf,np.inf)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('routine_name')
	parser.add_argument('hamiltonian_filename')
	parser.add_argument('n', type = int)
	parser.add_argument('-k', type = int, default = 2)#range of onebody_operators
	parser.add_argument('-H_k', type = int, default = 2)#size of "one-body terms" of hamiltonian
	parser.add_argument('-H_locality', default = 'short_range')#whether the hamiltonian is short- or long-range
	parser.add_argument('-H_range')#if longrange, can set the maximum range for hamiltonian terms
	parser.add_argument('-g', type = float)#coupling constant
	parser.add_argument('-Delta', type = float)#coupling constant
	parser.add_argument('-oo', '--obj_ord', type = int, default = 1)#order of norm used for minimization
	parser.add_argument('-mu', type = float, default = DEFAULT_MU)#parameter for learning
	parser.add_argument('-b', '--beta', type = float)#temperature
	parser.add_argument('-t', '--threshold', type = float, default = DEFAULT_THRESHOLD)#threshold for cutting off small eigenvalues of C
	parser.add_argument('-pe','--periodic', type = bool, default = False)
	parser.add_argument('-d','--disorder', type = float, default = 0)
	parser.add_argument('-pr','--printing',  action = 'store_true')
	parser.add_argument('-sp', '--skip_plotting', action = 'store_true')
	parser.add_argument('-nc', '--nochecks', action = 'store_true', help = 'skip some checks to speed it up')
	parser.add_argument('-cc', '--nocache', action = 'store_true', help = 'skip loading cached expectations')
	parser.add_argument('-ns', '--nosave', action = 'store_true', help = 'skip creating a run directory')
	parser.add_argument('-ow', '--overwrite_cache', action = 'store_true', help = 'overwrite existing cache of expectation values if it exists')
	args = parser.parse_args()

	if args.routine_name == 'testThermalStateLearning':
		testThermalStateLearning(args)
	elif args.routine_name == 'testGroundstateLearning':
		testGroundstateLearning(args)
	elif args.routine_name == 'testModularHamiltonianLearning':
		testModularHamiltonianLearning(args)
	elif args.routine_name == 'compareQuantities':
		C_a,eigvals_a,eigvecs_a,cutoff_a, onebody_terms_a, H_a, F_prime_a = compareQuantities(args)
		C_b,eigvals_b,eigvecs_b,cutoff_b, onebody_terms_b, H_b, F_prime_b = testThermalStateLearning(args)
		assert np.array_equal(C_a, C_b)
		assert np.array_equal(eigvals_a, eigvals_b)
		assert np.array_equal(eigvecs_a, eigvecs_b)
		assert np.array_equal(cutoff_a, cutoff_b)
		assert np.array_equal(onebody_terms_a, onebody_terms_b)
		assert H_a==H_b
		assert np.array_equal(F_prime_a.toNumpy(), F_prime_b.toNumpy())
	else:
		raise ValueError(f'unrecognized routine name {args.routine_name}')
	
