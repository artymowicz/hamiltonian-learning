import numpy as np
import utils
import itertools
from utils import multiplyPaulis
import time

#a tensor C_ijkl such that a_i[a_j,a_k] = sum_l C_ijkl b_l,
#where a_i, a_k are single-body operators, a_j is a "hamiltonian operator" and b_l are three-body operators
def buildTripleProductTensor(onebody_operators, hamiltonian_terms, printing = False):
	if printing:
		tprint('building triple product tensor')

	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

	utils.tprint(f'n = {n}, r = {r}, h = {h}')

	utils.tprint('creating noncommuting_dict')

	#create a dict that, given a hamiltonian operator, returns a list of 
	#all one-body operators that don't commute with it
	noncommuting_dict = {}
	for a in hamiltonian_terms:
		noncommuting_with_a = []
		for b in onebody_operators:
			if utils.checkCommute(a,b) == False:
				noncommuting_with_a.append(b)
		noncommuting_dict[a] = noncommuting_with_a

	#create a dict of indices of one-body operators
	onebody_operators_dict = utils.invertStringList(onebody_operators)

	indices = []
	values = []

	threebody_operators = []
	threebody_indices_dict = {}
	threebody_len = 0

	utils.tprint('beginning main loop')
	for i in range(r):
		for j in range(h):
			a = onebody_operators[i]
			b = hamiltonian_terms[j]
			V,v = multiplyPaulis(a,b)
			for c in noncommuting_dict[b]:
				
				W,w = multiplyPaulis(V,c)#v * w * W = abc which equals a[b,c]/2 since b and c are noncommuting paulis

				if W not in threebody_indices_dict:
					threebody_operators.append(W)
					threebody_len += 1
					l = threebody_len-1
					threebody_indices_dict[W] = l
				else:
					l = threebody_indices_dict[W]

				k = onebody_operators_dict[c]

				indices.append([i,j,k,l])
				values.append(2*v*w)
				
	shape = np.asarray([r,h,r,len(threebody_operators)])
	out = utils.SparseTensor(shape,indices,values)
	out.removeZeros()

	if printing:
		tprint(f'number of nonzeros in triple_product_tensor is {len(out.values)} ({len(values)} before removing spurious entries)')

	threebody_operators = np.asarray(threebody_operators, dtype = '<U' + str(n))
	threebody_operators.sort()

	return out, threebody_operators

pauli_char_to_int = dict(I = 0, X = 1, Y = 2, Z = 3)
def pauliStringToIntArray(p):
	return np.array([pauli_char_to_int[x] for x in p], dtype = 'uint8')

def buildThreebodyTermsFast(onebody_operators, hamiltonian_terms, printing = False):
	if printing:
		utils.tprint('building three body terms')

	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

	if printing:
		utils.tprint(f'n = {n}, r = {r}, h = {h}')

	onebody_operators = np.asarray(onebody_operators)
	hamiltonian_terms = np.asarray(hamiltonian_terms)

	#print(f'n = {n}, r = {r}, h = {h}')

	#noncommuting[i] is a numpy array containing all the indices of onebody operators that don't commute with hamiltonian_terms[i]
	noncommuting = [None]*h
	for i in range(h):
		a = hamiltonian_terms[i]
		noncommuting_with_a = []
		for j in range(r):
			b = onebody_operators[j]
			if utils.checkCommute(a,b) == False:
				noncommuting_with_a.append(j)
		noncommuting[i] = np.asarray(noncommuting_with_a, dtype='uint32')

	#utils.tprint(f'average number of noncommuting onebodys per hamiltonian term: {np.mean([len(noncommuting[i]) for i in range(h)])}')
	#utils.tprint(f'max noncommuting onebodys per hamiltonian term: {max([len(noncommuting[i]) for i in range(h)])}')

	onebody_operators_intarray = np.asarray([pauliStringToIntArray(p) for p in onebody_operators], dtype = 'uint8')
	hamiltonian_terms_intarray = np.asarray([pauliStringToIntArray(p) for p in hamiltonian_terms], dtype = 'uint8')

	out_intarray = np.empty(shape = (0,n), dtype = 'uint8')
	out_set = set()
	for j in range(h):
		b = hamiltonian_terms_intarray[j]
		for k in noncommuting[j]:
			c = onebody_operators_intarray[k]
			bc = np.bitwise_xor(b,c)
			new_terms_intarray = np.bitwise_xor(onebody_operators_intarray,bc)
			new_terms_chararray = np.asarray(['I','X','Y','Z'])[new_terms_intarray]
			l = new_terms_chararray.shape[0]
			out_set.update(new_terms_chararray.reshape(n*l).view('<U'+str(n)))

	out = np.asarray(list(out_set))
	out.sort()
	return out

def sewPairs(a,b):
	m = a.shape[0]
	n = b.shape[0]
	x = np.repeat(a, n, axis=0)
	y = np.tile(b, (m, 1))
	#print(f'a.shape = {a.shape}')
	#print(f'b.shape = {b.shape}')
	#print(f'x.shape = {x.shape}')
	#print(f'y.shape = {y.shape}')
	#return np.hstack((x.reshape((m*n,1)),y))
	return np.hstack((x,y))

def buildTripleProductTensorFast(onebody_operators, hamiltonian_terms, threebody_operators, printing = False):
	if printing:
		utils.tprint('building triple product tensor')

	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

	if printing:
		utils.tprint(f'n = {n}, r = {r}, h = {h}')

	onebody_operators = np.asarray(onebody_operators)
	hamiltonian_terms = np.asarray(hamiltonian_terms)

	noncommuting = [None]*h
	for i in range(h):
		a = hamiltonian_terms[i]
		noncommuting_with_a = []
		for j in range(r):
			b = onebody_operators[j]
			if utils.checkCommute(a,b) == False:
				noncommuting_with_a.append(j)
		noncommuting[i] = np.asarray(noncommuting_with_a, dtype='int32')

	onebody_operators_intarray = np.asarray([pauliStringToIntArray(p) for p in onebody_operators], dtype = 'uint8')
	hamiltonian_terms_intarray = np.asarray([pauliStringToIntArray(p) for p in hamiltonian_terms], dtype = 'uint8')

	threebody_operators_indices = dict(zip(threebody_operators, range(len(threebody_operators))))

	##phase_table[4*x+y] = phase of sigma_x*sigma_y, where x and y are integers between 0 and 3 representing two paulis
	phase_table = np.zeros(16, dtype = 'uint8')
	phase_table[[6,11,13]] = 1
	phase_table[[7,9,14]] = 3

	###first compute the commutators [a_j, a_k] and put them into an array whose rows are [j,k, (log coefficient of [a_j,a_k]), pauli of [a_j,a_k] as a list of ints]
	#utils.tprint('creating commutators array')	
	commutators_jk = []
	commutators_logz = []
	commutators_paulis = []
	for j in range(h):
		b = hamiltonian_terms_intarray[j]
		for k in noncommuting[j]:
			c = onebody_operators_intarray[k]
			commutators_jk.append([j,k])
			commutators_logz.append(np.mod(np.sum(phase_table[4*b+c]),4))
			commutators_paulis.append(np.bitwise_xor(b, c))

	commutators_jk = np.array(commutators_jk, dtype = 'uint32')
	commutators_logz = np.array(commutators_logz, dtype = 'uint8')
	commutators_paulis = np.array(commutators_paulis, dtype = 'uint8')

	combined_indices = sewPairs(np.arange(r).reshape((r,1)),commutators_jk)
	combined_paulis = sewPairs(onebody_operators_intarray, commutators_paulis)
	multiplied_paulis = np.bitwise_xor(combined_paulis[:,:n], combined_paulis[:,n:])
	partial_log_phases = np.sum(phase_table[4*combined_paulis[:,:n] + combined_paulis[:,n:]], axis = 1)
	total_log_phases = np.mod(partial_log_phases + np.tile(commutators_logz, r),4)
	phase_exp_table = np.array([1., 1j, -1., -1j])
	total_phases = 2*phase_exp_table[total_log_phases]

	t1 = time.time()
	pauli_char_table = np.array(['I','X','Y','Z'], dtype = '<U1')
	last_index = pauli_char_table[multiplied_paulis].view(dtype = '<U'+str(n))
	last_index = np.vectorize(threebody_operators_indices.__getitem__)(last_index) ### this is the current bottleneck
	t2 = time.time()
	if printing:
		utils.tprint(f'reversing l index list took {t2-t1:.2f} seconds')

	shape = np.asarray([r,h,r,len(threebody_operators)])
	#print(f'combined_indices.shape = {combined_indices.shape}')
	#print(f'last_index.shape = {last_index.shape}')
	#print(f'total_phases.shape = {total_phases.shape}')
	indices = np.hstack((combined_indices, last_index))
	out = utils.SparseTensor(shape,indices,total_phases)

	return out

def slow_fast_comparison(n,onebody_operators,hamiltonian_terms):
	t1 = time.time()
	threebody_operators = buildThreeBodyTermsFast(onebody_operators, hamiltonian_terms)
	t2 = time.time()
	triple_product_tensor_slow, threebody_operators_slow = utils.buildTripleProductTensor(onebody_operators, hamiltonian_terms)
	triple_product_tensor_slow.order()
	t3 = time.time()
	print(f'threebody operators equality check: {np.array_equal(threebody_operators, threebody_operators_slow)}')
	t4 = time.time()
	triple_product_tensor = buildTripleProductTensorFast(onebody_operators, hamiltonian_terms, threebody_operators_slow)
	t5 = time.time()
	if not triple_product_tensor.isEqual(triple_product_tensor_slow):
		print(f'triple product tensor equality check failed')
		print(f'triple_product_tensor_slow.shape = {triple_product_tensor_slow.shape}, triple_product_tensor.shape = {triple_product_tensor.shape}')
		print(triple_product_tensor.indices)
		print(triple_product_tensor_slow.indices)
		print(f'indices equality check: {np.array_equal(triple_product_tensor.indices,triple_product_tensor_slow.indices)}')
		print(f'values equality check: {np.array_equal(triple_product_tensor.indices,triple_product_tensor_slow.indices)}')
		#for i in range(len(triple_product_tensor.values)):
		#	print(f'{triple_product_tensor.indices[i]} {triple_product_tensor.values[i]} {triple_product_tensor_slow.values[i]}')
	else:
		print(f'triple product tensor equality check passed')

	#print(len(threebody_operators_old))
	#for i in range(len(threebody_operators_old)):
	#	print(f'{threebody_operators_new[i]}  {threebody_operators_old[i]}' )
	#for i in range(len(threebody_operators_old), len(threebody_operators_new)):
	#	print(threebody_operators_new[i])

	#utils.tprint(f'number of threebody operators = {len(threebody_operators_new)}')
	#print(np.array_equal(threebody_operators_old,threebody_operators_new))
	print(f'n = {n} number of threebody operators = {len(threebody_operators)}')
	print(f'computing threebody operators took {t2-t1:.2f} seconds')
	print(f'computing triple product tensor the slow way took {t3-t2:.2f} seconds')
	print(f'computing triple product tensor the fast way took {t5-t4:.2f} seconds')

def fast_time(n,onebody_operators,hamiltonian_terms):
	t1 = time.time()
	threebody_operators = utils.buildThreBbodyTermsFast(onebody_operators, hamiltonian_terms, printing = True)
	t2 = time.time()
	triple_product_tensor = utils.buildTripleProductTensorFast(onebody_operators, hamiltonian_terms, threebody_operators, printing = True)
	t3 = time.time()
	
	print(f'n = {n} number of threebody operators = {len(threebody_operators)}')
	print(f'computing threebody operators took {t2-t1:.2f} seconds')
	print(f'computing triple product tensor took {t3-t2:.2f} seconds')

if __name__ == '__main__':
	n = 10
	k = 2

	onebody_operators = utils.buildKLocalPaulis1D(n,k, False)
	hamiltonian_terms = utils.buildKLocalPaulis1D(n,k, False)
	#hamiltonian_terms = utils.buildKLocalCorrelators1D(n, 1, False, d_max = 5)
	fast_time(n,onebody_operators,hamiltonian_terms)




