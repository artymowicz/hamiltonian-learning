import tenpy
import numpy as np
import state_simulator
import hamiltonian_learning
import hamiltonian_learning_tester
import argparse
import utils
import time
from multiprocessing import Pool
import functools

DEFAULT_THRESHOLD = 1e-9
DEFAULT_MU = 0

my_to_tenpy = dict(I = 'Id', X = 'Sigmax', Y = 'Sigmay', Z = 'Sigmaz')

pauli_generators = {'X': np.array([[0,1],[1,0]], dtype = complex),
					'Y': np.array([[0,0.-1.j],[0.+1.j,0]]),
					'Z': np.array([[1.,0.],[0,-1.]], dtype=complex),
					'I': np.identity(2, dtype=complex)}

def DFScomputeParallel(n, psi, operators, n_threads, printing = False):
	l = len(operators)
	out = np.zeros(l)
	ind_list = [(l//n_threads)*k for k in range(n_threads)]
	ind_list += [l]
	operators_chunks = [operators[ind_list[i]:ind_list[i+1]] for i in range(n_threads)]
	#f = lambda x : DFScomputeNew(n, psi, x, printing = printing)
	f = functools.partial(DFScomputeNew, n, psi)
	#processes = [Process(target = DFScomputeNew, args = (n, psi, chunk, False)) for chunk in operators_chunks]
	with Pool(n_threads) as p:
		results_chunks = p.map(f, operators_chunks)
	out = np.concatenate(tuple(results_chunks))
	return out

def DFScomputeNew(n, psi, operators, printing = False):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*(n+1)
	L_tensors[0] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vR', 'vR*'])

	tenpy_calls = 0

	print(f'running DFSComputeNew with n = {n}, l = {l}')

	###compute R_tensors
	R_tensors = [None]*(n+1)
	R_tensors[n] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vL', 'vL*'])

	for k in range(n,0,-1):
		R = R_tensors[k]
		B = psi.get_B(k-1)
		tensor_list = [R,B, B.conj()]
		tensor_names = ['R', 'B','B*']
		leg_contractions = [['R', 'vL', 'B', 'vR'], ['R', 'vL*', 'B*', 'vR*']]
		leg_contractions += [['B', 'p', 'B*', 'p*']]
		open_legs = [['B', 'vL', 'vL'], ['B*', 'vL*', 'vL*']]
		R_tensors[k-1] = tenpy.algorithms.network_contractor.contract(tensor_list, tensor_names, leg_contractions, open_legs)
		#tenpy_calls += 1

	for i in range(l):
		if i%(l//100) == 0:
			utils.tprint(f'{i/l:.0%} done')
		p = operators[i]
		#j = utils.firstDifferingIndex(p, previous)

		#get the first index that differs from the previous p
		for j in range(n):
			if p[j] != previous[j]:
				break

		#get the last nontrivial index of p (defaults to -1 if p is the identity)
		last_nontriv = -1
		for x in range(n):
			if p[x] != 'I':
				last_nontriv = x

		y = max(j,x)

		for k in range(j,y+1):
				L = L_tensors[k]
				B = psi.get_B(k)
				onsite_term = psi.sites[k].get_op(my_to_tenpy[p[k]])
				tensor_list = [L, B, B.conj(), onsite_term]
				tensor_names = ['L', 'B','B*', 'O']
				leg_contractions = [['L', 'vR', 'B', 'vL'], ['L', 'vR*', 'B*', 'vL*']]
				leg_contractions += [['B', 'p', 'O', 'p']]
				leg_contractions += [['B*', 'p*', 'O', 'p*']]
				open_legs = [['B', 'vR', 'vR'], ['B*', 'vR*', 'vR*']]
				L_tensors[k+1] = tenpy.algorithms.network_contractor.contract(tensor_list, tensor_names, leg_contractions, open_legs)
				tenpy_calls += 1

		out[i] = np.real(tenpy.algorithms.network_contractor.contract([L_tensors[y+1], R_tensors[y+1]], ['L', 'R'], [['L','vR','R','vL'], ['L','vR*','R','vL*']]))
		tenpy_calls += 1
		previous = p

	print(f'total number of contractions = {tenpy_calls}')
	print(f'{tenpy_calls/l - 1:.1f} nontrivial contractions per evaluation')
	return out

def DFScomputeOld(n, psi, operators, printing = False):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*n
	tenpy_calls = 0

	print(f'running DFSComputeOld with n = {n}, l = {l}')


	###computing R tensors
	L_tensors = [None]*(n+1)

	for i in range(l):
		if i%(l//100) == 0:
			utils.tprint(f'{i/l:.0%} done')
		p = operators[i]
		#j = utils.firstDifferingIndex(p, previous)

		#get the first differing index
		for j in range(n):
			if p[j] != previous[j]:
				break
		
		#last nontrivial index
		z = -1
		for x in range(n):
			if p[x] != 'I':
				z = x

		for k in range(j,n):
			if k == 0:
				B = psi.get_B(k)
				onsite_term = psi.sites[k].get_op(my_to_tenpy[p[k]])
				tensor_list = [B, B.conj(), onsite_term]
				tensor_names = ['B','B*','O']
				leg_contractions = [['B', 'vL', 'B*', 'vL*']] #contract trivial edge virtual index
				leg_contractions += [['B', 'p', 'O', 'p']]
				leg_contractions += [['B*', 'p*', 'O', 'p*']]
				open_legs = [['B', 'vR', 'vR'], ['B*', 'vR*', 'vR*']]
				L_tensors[k] = tenpy.algorithms.network_contractor.contract(tensor_list, tensor_names, leg_contractions, open_legs)
				tenpy_calls += 1
			else:
				L = L_tensors[k-1]
				B = psi.get_B(k)
				onsite_term = psi.sites[k].get_op(my_to_tenpy[p[k]])
				tensor_list = [L, B, B.conj(), onsite_term]
				tensor_names = ['L', 'B','B*', 'O']
				leg_contractions = [['L', 'vR', 'B', 'vL'], ['L', 'vR*', 'B*', 'vL*']]
				leg_contractions += [['B', 'p', 'O', 'p']]
				leg_contractions += [['B*', 'p*', 'O', 'p*']]
				open_legs = [['B', 'vR', 'vR'], ['B*', 'vR*', 'vR*']]
				L_tensors[k] = tenpy.algorithms.network_contractor.contract(tensor_list, tensor_names, leg_contractions, open_legs)
				tenpy_calls += 1

		out[i] = np.real(tenpy.algorithms.network_contractor.contract([L_tensors[n-1]], ['L'], [['L', 'vR','L','vR*']]))
		tenpy_calls += 1
		previous = p
	print(f'total number of contractions = {tenpy_calls}')
	print(f'{tenpy_calls/l - 1:.1f} nontrivial contractions per evaluation')
	return out

def DFScomputeFromPureState(n, psi, operators):
	l = len(operators)
	out = np.zeros(l, dtype = complex)
	previous = '-'*n
	L_tensors = [None]*n
	Bs = [psi.get_B(k).to_ndarray() for k in range(n)]
	Bstars = [np.conjugate(B) for B in Bs]

	print(f'running DFSCompute with n = {n}, l = {l}')

	t1 = time.time()
	contractions = 0

	for i in range(l):
		p = operators[i]
		#j = utils.firstDifferingIndex(p, previous)

		#get the first differing index
		for j in range(n):
			if p[j] != previous[j]:
				break

		if i%(l//100) == 0:
			utils.tprint(f'{i/l:.0%} done')

		for k in range(j,n):
			if k == 0:
				B = Bs[k]
				Bstar = Bstars[k]
				O = pauli_generators[p[k]]
				L_tensors[k] = np.einsum('ijk,jl,ilm->km',B,O,Bstar)
				contractions += 1
			else:
				L = L_tensors[k-1]
				B = Bs[k]
				Bstar = Bstars[k]
				O = pauli_generators[p[k]]
				L_tensors[k] = np.einsum('ij,ikl,km,jmn->ln',L,B,O,Bstar)
				contractions += 1

		out[i] = L_tensors[n-1][0,0]
		previous = p

	t2 = time.time()

	print(f'ran DFSCompute with n = {n}, l = {l}, compute time = {t2-t1:.2f} seconds')
	print(f'total number of contractions = {contractions}')
	print(f'{contractions/l:.1f} contractions per evaluation')

	return np.real(out)

def naiveCompute(n,psi, operators):
	l = len(operators)
	out = np.zeros(l)
	for (i,p) in zip(range(l), operators):
		p_term = state_simulator.pauliStringToTenpyTerm(n,p)
		out[i] = psi.expectation_value_term(p_term)
	return out

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

	H, onebody_operators, hamiltonian_terms, simulator_params, learning_params = hamiltonian_learning_tester.parseCommandLineInput(args)
	n = H.n
	utils.tprint('generating threebody operators')
	_, threebody_operators = utils.buildTripleProductTensor(onebody_operators, hamiltonian_terms)
	utils.tprint(f'number of threebody operators is {len(threebody_operators)}')
	utils.tprint('sorting threebody operators')
	threebody_operators = sorted(threebody_operators, reverse = True)
	utils.tprint('computing state')
	psi = state_simulator.computeGroundstateDMRG(H, simulator_params)
	#psi = state_simulator.computeThermalStateByPurification(H, args.beta, simulator_params)

	'''
	B = psi.get_B(1)
	print(B)
	print(B.conj())
	assert False
	'''

	#psi = state_simulator.computeThermalStateByPurification(H, args.beta, simulator_params)
	utils.tprint('computing threebody expectations using DFS')
	threebody_expectations_DFS = DFScomputeParallel(n,psi, threebody_operators, n_threads = 4)
	#threebody_expectations_DFS = DFScomputeFromPureState(n,psi, threebody_operators)

	assert False

	utils.tprint('computing threebody expectations naively')
	threebody_expectations_naive = naiveCompute(n,psi, threebody_operators)
	#threebody_expectations_naive = np.zeros(len(threebody_expectations_DFS))

	print(f'difference between naive and DFS = {np.linalg.norm(threebody_expectations_DFS - threebody_expectations_naive)}')

	#print()
	#for i in range(len(threebody_operators)):
	#	print(f'{threebody_operators[i]} : {threebody_expectations_DFS[i]:+f} : {threebody_expectations_naive[i]:+f}')
	#print()

