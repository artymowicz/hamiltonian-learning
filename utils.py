import numpy as np
import scipy
import itertools
import functools as ft
import datetime
import os
import time


class SparseTensor:
	def __init__(self, shape, indices, values, order = False):
		self.shape = np.asarray(shape, dtype = int)
		self.indices = np.asarray(indices, dtype = int)
		self.values = np.asarray(values)
		assert self.indices.shape == (len(self.values), len(self.shape))
		for i in range(len(shape)):
			assert max(self.indices[:,i])<self.shape[i]
		if order:
			self.order()

	def __str__(self):
		return f'SparseTensor of shape {self.shape} with {len(self.values)} stored elements'

	def isEqual(self,other):
		return np.array_equal(self.shape, other.shape) and np.array_equal(self.indices, other.indices) and np.array_equal(self.values, other.values)

	#orders its entries by lexicographical order of index
	def order(self, little_endian = False):
		if little_endian:
			nonzeros_indices_sorted = sorted(range(len(self.values)), key = lambda i: list(self.indices[i]).reverse())
		else:
			nonzeros_indices_sorted = sorted(range(len(self.values)), key = lambda i: list(self.indices[i]))
		self.indices = self.indices[nonzeros_indices_sorted]
		self.values = self.values[nonzeros_indices_sorted]

	def removeZeros(self):
		nonzeros = []
		for i in range(len(self.values)):
			if self.values[i] != 0:
				nonzeros.append(i)
		self.indices = self.indices[nonzeros]
		self.values = self.values[nonzeros]

	def toScipySparse(self):
		assert len(self.shape) == 2
		return scipy.sparse.coo_array((self.values,(self.indices[:,0],self.indices[:,1])), shape = self.shape)

	def toNumpy(self):
		out = np.zeros(self.shape, dtype = complex)
		for i in range(len(self.values)):
			out[tuple(self.indices[i])] = self.values[i]
		return out

	def transpose(self, permutation):
		assert len(permutation) == len(self.shape)
		assert sorted(permutation) == list(range(len(self.shape)))
		self.shape = self.shape[permutation]
		self.indices = self.indices[:,permutation]

	#vectorizes given axes in Fortran order. The new axis is by default the leftmost one
	def vectorize(self, axes):
		axes_complement = [i for i in range(len(self.shape)) if i not in axes]
		prefactors = [1]
		for i in range(len(axes)-1):
			prefactors.append(prefactors[i]*self.shape[axes[i+1]])
		def indexVectorization(index_in):
			first_index_out = sum([prefactors[i]*index_in[axes[i]] for i in range(len(axes))])
			return np.concatenate(([first_index_out], index_in[axes_complement]))
		indices_new = []
		for index in self.indices:
			indices_new.append(indexVectorization(index))
		if len(axes_complement) > 0:
			shape_new = np.concatenate(([np.prod(self.shape[axes])], self.shape[axes_complement]))
		else:
			shape_new = np.prod(self.shape[axes]).reshape((1,))
		self.shape = shape_new
		self.indices = np.asarray(indices_new)

	#devectorizes axis given by vector_axis. Expanded axes become the leftmost ones
	def devectorize(self, vector_axis, new_axes_shape):
		assert self.shape[vector_axis] == np.prod(new_axes_shape)
		axes_complement = [i for i in range(len(self.shape)) if i != vector_axis]
		self.shape = np.concatenate((new_axes_shape,self.shape[axes_complement]))

		def indexDevectorization(index_in):
			index_out = np.zeros(len(self.shape), dtype = int)
			x = index_in[vector_axis]
			for i in range(len(new_axes_shape)-1):
				index_out[i] = x%new_axes_shape[i]
				x = x//new_axes_shape[i]
			index_out[len(new_axes_shape)-1] = x
			index_out[len(new_axes_shape):] = index_in[axes_complement]
			return index_out

		indices_new = np.zeros((len(self.values), len(self.shape)), dtype = int)

		for i in range(len(self.indices)):
			indices_new[i] = indexDevectorization(self.indices[i])
		
		self.indices = indices_new

	#Contracts with a dense vector v along the last axis. For instance if A is order 3, returns sum_k A_ijk v_k
	#If skip_ordering = True, assumes that indices are lexicographically (ie. big-endian) ordered
	def contractRight(self,v, skip_ordering = False):
		if self.shape[-1] != len(v):
			raise ValueError(f'rightmost tensor dimension {self.shape[-1]} does not match vector dimension {len(v)}')

		if not skip_ordering:
			self.order()
		out_shape = self.shape[:-1]
		out_indices = []
		out_values = []

		current_sum = 0
		
		for i in range(len(self.values)):
			current_index = self.indices[i]
			current_sum += self.values[i]*v[current_index[-1]]
			if i == len(self.values)-1:
				out_values.append(current_sum)
				out_indices.append(current_index[:-1])
			else:
				next_index = self.indices[i+1]
				if any(next_index[:-1] != current_index[:-1]):
					out_values.append(current_sum)
					out_indices.append(current_index[:-1])
					current_sum = 0

		out = SparseTensor(out_shape, out_indices, out_values, order = False)
		out.removeZeros()
		
		return out

	#Contracts with a dense vector v along the first axis. For instance if A is order 3, returns sum_i A_ijk v_i
	#If skip_ordering = True, assumes that indices are REVERSE-lexicographically (ie. little-endian) ordered
	def contractLeft(self,v, skip_ordering = False):
		if self.shape[0] != len(v):
			raise ValueError(f'leftmost tensor dimension {self.shape[0]} does not match vector dimension {len(v)}')

		assert self.shape[-1] == len(v)
		if not skip_ordering:
			self.order(little_endian = True)
		out_shape = self.shape[1:]
		out_indices = []
		out_values = []

		current_sum = 0
		
		for i in range(len(self.values)):
			current_index = self.indices[i]
			current_sum += self.values[i]*v[current_index[0]]
			if i == len(self.values)-1:
				out_values.append(current_sum)
				out_indices.append(current_index[1:])
			else:
				next_index = self.indices[i+1]
				if any(next_index[1:] != current_index[1:]):
					out_values.append(current_sum)
					out_indices.append(current_index[1:])
					current_sum = 0

		out = SparseTensor(out_shape, out_indices, out_values, order = False)
		out.removeZeros()
		
		return out

#dict containing the pauli and phase resulting from multiplying two paulis
single_pauli_mult_matrix = {}

single_pauli_mult_matrix[('I','I')] = ('I', 1.+0.j)
single_pauli_mult_matrix[('I','X')] = ('X', 1.+0.j)
single_pauli_mult_matrix[('I','Y')] = ('Y', 1.+0.j)
single_pauli_mult_matrix[('I','Z')] = ('Z', 1.+0.j)

single_pauli_mult_matrix[('X','I')] = ('X', 1.+0.j)
single_pauli_mult_matrix[('X','X')] = ('I', 1.+0.j)
single_pauli_mult_matrix[('X','Y')] = ('Z', 0.+1.j)
single_pauli_mult_matrix[('X','Z')] = ('Y', 0.-1.j)

single_pauli_mult_matrix[('Y','I')] = ('Y', 1.+0.j)
single_pauli_mult_matrix[('Y','X')] = ('Z', 0.-1.j)
single_pauli_mult_matrix[('Y','Y')] = ('I', 1.+0.j)
single_pauli_mult_matrix[('Y','Z')] = ('X', 0.+1.j)

single_pauli_mult_matrix[('Z','I')] = ('Z', 1.+0.j)
single_pauli_mult_matrix[('Z','X')] = ('Y', 0.+1.j)
single_pauli_mult_matrix[('Z','Y')] = ('X', 0.-1.j)
single_pauli_mult_matrix[('Z','Z')] = ('I', 1.+0.j)

def multiplyPaulis(pauli_1,pauli_2):
	assert len(pauli_1) == len(pauli_2)
	pauli_out = ''
	phase = 1. + 0.j
	for i in range(len(pauli_1)):
		W, z = single_pauli_mult_matrix[(pauli_1[i], pauli_2[i])]
		pauli_out = pauli_out + W
		phase = phase*z
	return pauli_out, phase

def checkCommute(pauli_1,pauli_2):
	assert len(pauli_1) == len(pauli_2)
	total = 0
	for i in range(len(pauli_1)):
		a = pauli_1[i]
		b = pauli_2[i]
		if a == 'I' or b =='I' or a == b:
			pass
		else:
			total += 1
	return total%2 == 0

def determineSupport(pauli_string):
	return [c != 'I' for c in pauli_string]

def supportBitString(pauli):
	return ''.join(['0' if x == 'I' else '1' for x in pauli])

def weight(pauli_string):
	return sum([1 for c in pauli_string if c != 'I'])

#given a list of unique strings, returns a dict that takes a string and returns the index of that string
def invertStringList(l):
	d = {}
	for i in range(len(l)):
		d[l[i]]= i
	return d

#returns compressed representation of a pauli.
#first number is the Hamming weight
#then, for single-site pauli factor, we have the letter X,Y,or Z and the index of the corresponding site.
#For instance IIYIIX is converted to 2 Y 2 X 5
def compressPauli(pauli):
	n = len(pauli)
	supp = determineSupport(pauli)
	w = sum(supp)
	out = str(w)
	for i in range(n):
		if supp[i]:
			out += ' '+pauli[i] + ' ' + str(i)
	return out

def compressPauliToList(pauli):
	n = len(pauli)
	supp = determineSupport(pauli)
	w = sum(supp)
	out = [w]
	for i in range(n):
		if supp[i]:
			out.append((pauli[i],i))
	return out

#returns decompressed representation of a pauli. For instance 2 Y 2 X 5 is converted to IIYIIX 
#TODO: test
def decompressPauli(pauli_compressed,n):
	pauli_list_out = ['I']*n
	pauli_compressed_list = pauli_compressed.split(' ')
	k = int(pauli_compressed_list[0])
	for i in range(k):
		pauli_char = pauli_compressed_list[1+2*i]
		index = int(pauli_compressed_list[2+2*i])
		pauli_list_out[index] = pauli_char
	return ''.join(pauli_list_out)

#returns a list of paulis of Hamming weight w
def buildWeightWPaulis(n,w):
	if w == 0:
		return ['I'*n]
	out = []
	for c in itertools.combinations(range(n),w):
		for indices in itertools.product(range(3), repeat = w):
			p = ['I',]*n
			for i in range(w):
				p[c[i]] = ['X','Y','Z'][indices[i]]
			out.append("".join(p))
	return out

#returns a list of paulis up to and including weight w
def buildPaulisUpToWeightW(n,w):
	out = []
	for i in range(0,w+1):
		out = out + buildWeightWPaulis(n,i)
	assert out[0] == 'I'*n
	return out[1:]

def buildKLocalPaulis1D(n,k, periodic_bc):
	if n == 1:
		if k == 0:
			return ['I']
		else:
			return ['I','X','Y','Z']
	out = set()
	if periodic_bc:
		for i in range(n):
			for indices in itertools.product(range(4), repeat = k):
				p = ['I',]*n
				for j in range(k):
					p[(i+j)%n] = ['X','Y','Z','I'][indices[j]]
				out.add("".join(p))
	else:
		for i in range(n-k+1):
			for indices in itertools.product(range(4), repeat = k):
				p = ['I',]*n
				for j in range(k):
					p[i+j] = ['X','Y','Z','I'][indices[j]]
				out.add("".join(p))

	out = sorted(list(out))
	assert out[0] == 'I'*n
	return out

def buildKLocalCorrelators1D(n,k, periodic_bc, d_max = None):
	if d_max is None:
		d_max = n
	out_set = set()

	if periodic_bc is False:
		for i in range(n-k+1):
			upper_range = min(n-k+1, i+k+d_max+1)
			for j in range(i+k, upper_range):
				for indices in itertools.product(range(4), repeat = 2*k):
					p = ['I',]*n
					for l in range(k):
						p[i+l] = ['X','Y','Z','I'][indices[l]]
						p[j+l] = ['X','Y','Z','I'][indices[k+l]]
					out_set.add(''.join(p))
	else:
		raise ValueError('havent implemented perioidc bc here')

	out = sorted(list(out_set))
	return out

def embedPauli(n, p,region):
	out_list = ['I']*n
	for i in range(len(region)):
		out_list[region[i]] = p[i]
	return ''.join(out_list)

#given a list of operators it returns a list of those that are localized within the region
#region is a list of indices
def restrictOperators(n, operators, region):
	region_complement = [i for i in range(n) if i not in region]
	out = []
	for p in operators:
		support = determineSupport(p)
		if not any([support[i] for i in region_complement]):
			p_restricted = ''.join([p[i] for i in region])
			out.append(p_restricted)
	return out

#single-qubit cliffords
clifford_generators = {'X': np.array([[1,1],[-1,1]])/np.sqrt(2),
						'Y': np.array([[1,-1j],[-1j,1]])/np.sqrt(2), 
						'Z': np.array([[1-1j,0],[0,1+1j]])/np.sqrt(2), 
						'I': np.identity(2, dtype=complex)}

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

def generateCliffordMatrix(pauli_string):
	pauli_list = [clifford_generators[c] for c in pauli_string]
	return ft.reduce(scipy.sparse.kron, pauli_list) #TODO: see if we're using dense matrices anywhere

def intToBitstring(value, length):
	return format(value, '0{}b'.format(length))

def intToBoolstring(value, length):
	return bitStringToBoolstring(intToBitstring(value,length),length)

def bitStringToBoolstring(bitstring, length):
	return [c == '1' for c in bitstring]

#loads a wavefunction (as a numpy array) from a filename
def loadWavefunction(state_filename):
	with open('./states/'+state_filename+'.txt','r') as f:
		lines = f.readlines()
	l = len(lines)
	n = int(lines[0][:-1])
	assert l == 2**n+1
	psi = np.zeros(2**n, dtype = complex)
	for i in range(2**n):
		line = lines[i+1]
		re_str,im_str = line[:-1].split(' ')#last character of each line is \n
		psi[i] = float(re_str)+ 1j*float(im_str)
	return n,psi

def strToBool(s):
	if s == 'True' or s == 'T' or s == 't' or s == '1':
		return True
	elif s == 'False' or s == 'F' or s == 'f' or s == '0':
		return False
	else:
		raise ValueError

#assumes a and b are same length
def firstDifferingIndex(a,b):
	for i in range(len(a)):
		if a[i] != b[i]:
			return i
	raise ValueError('inputs are equal or second one is longer')

#returns a function that takes a list of Paulis and returns their expectations
def buildStateEvaluator(state_filename, type):
	if type == 'wavefunction':
		n,psi = loadWavefunction(state_filename)
		evaluator = lambda paulis : [computeExpectationWaveFunc(p, psi, n) for p in paulis]
	elif type == 'MPS':
		#TODO: implement MPS
		raise ValueError
	else:
		raise ValueError
	return evaluator, n

#builds tensor C_ijk such that a_ia_j = sum_k C_ijk a_k
def buildMultiplicationTensor(onebody_operators):
	n = len(onebody_operators[0])

	R = len(onebody_operators)
	indices = []
	values = []

	twobody_operators = []
	twobody_indices_dict = {}
	l = 0

	for i in range(R):
			for j in range(i,R):
				W,z = multiplyPaulis(onebody_operators[i],onebody_operators[j])

				if W not in twobody_indices_dict:
					twobody_operators.append(W)
					l += 1
					twobody_indices_dict[W] = l-1

				indices.append([i,j,twobody_indices_dict[W]])
				values.append(z)
				if j > i:
					indices.append([j,i,twobody_indices_dict[W]])
					values.append(np.conjugate(z))

	mult_tensor = SparseTensor([R,R,l], indices, values)
	return mult_tensor, twobody_operators

pauli_char_to_int = dict(I = 0, X = 1, Y = 2, Z = 3)
def pauliStringToIntArray(p):
	return np.array([pauli_char_to_int[x] for x in p], dtype = 'uint8')

def buildThreeBodyTermsFast(onebody_operators, hamiltonian_terms, printing = False):
	if printing:
		tprint('building three body terms')

	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

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
			if checkCommute(a,b) == False:
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
		tprint('building triple product tensor')

	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

	if printing:
		tprint(f'n = {n}, r = {r}, h = {h}')

	onebody_operators = np.asarray(onebody_operators)
	hamiltonian_terms = np.asarray(hamiltonian_terms)

	noncommuting = [None]*h
	for i in range(h):
		a = hamiltonian_terms[i]
		noncommuting_with_a = []
		for j in range(r):
			b = onebody_operators[j]
			if checkCommute(a,b) == False:
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
		tprint(f'reversing l index list took {t2-t1:.2f} seconds')

	shape = np.asarray([r,h,r,len(threebody_operators)])
	#print(f'combined_indices.shape = {combined_indices.shape}')
	#print(f'last_index.shape = {last_index.shape}')
	#print(f'total_phases.shape = {total_phases.shape}')
	indices = np.hstack((combined_indices, last_index))

	out = SparseTensor(shape,indices,total_phases)## this should be sorted

	return out

#a tensor C_ijkl such that a_i[a_j,a_k] = sum_l C_ijkl b_l,
#where a_i, a_k are single-body operators, a_j is a "hamiltonian operator" and b_l are three-body operators
def buildTripleProductTensor(onebody_operators, hamiltonian_terms, printing = False):
	if printing:
		tprint('building triple product tensor')

	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

	#create a dict that, given a hamiltonian operator, returns a list of 
	#all one-body operators that don't commute with it
	noncommuting_dict = {}
	for a in hamiltonian_terms:
		noncommuting_with_a = []
		for b in onebody_operators:
			if checkCommute(a,b) == False:
				noncommuting_with_a.append(b)
		noncommuting_dict[a] = noncommuting_with_a

	#create a dict of indices of one-body operators
	onebody_operators_dict = invertStringList(onebody_operators)

	indices = []
	values = []

	threebody_operators = []
	threebody_indices_dict = {}
	threebody_len = 0

	for i in range(r):
		for j in range(h):
			a = onebody_operators[i]
			b = hamiltonian_terms[j]
			for c in noncommuting_dict[b]:
				V,v = multiplyPaulis(a,b)
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
	out = SparseTensor(shape,indices,values)
	out.removeZeros()
	if printing:
		tprint(f'number of nonzeros in triple_product_tensor is {len(out.values)} ({len(values)} before removing spurious entries)')

	return out, threebody_operators

def tprint(s):
	current_time_string = time.strftime("%H:%M:%S", time.localtime())
	print(f'{current_time_string}: {s}')

##### LOADING AND SAVING

#creates a directory to save the results of the run
def createSaveDirectory():
	now = datetime.datetime.now()
	dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
	if not os.path.exists('./runs/'):
		os.mkdir('./runs/')
	dirname = f'./runs/{dt_string}'
	os.mkdir(dirname)
	return dirname

#DEPRECATED
def saveExpectations(observables_list, expectations_list, filename):
	with open('./expectations/'+ filename, 'w') as f:  
	    f.write(','.join(['pauli', 'pauli_compressed' ,'expectation'])+'\n')
	    l = len(observables_list)
	    assert len(expectations_list) == l
	    for i in range(l):
	    	f.write(','.join([observables_list[i], compressPauli(observables_list[i]), str(expectations_list[i])])+'\n')

#DEPRECATED
def loadExpectations(filename):
	with open('./expectations/'+filename,'r') as f:
		lines = f.readlines()
	l = len(lines)
	assert lines[0].split(',')[0] == 'pauli'
	print(lines[0].split(',')[2])
	assert lines[0].split(',')[2] == 'expectation\n'

	paulis = []
	expectations = []
	for line in lines[1:]:
		line_split = line.split(',')
		pauli = line_split[0]
		expectation_string = line_split[2][:-1]#line ends in \n so this needs to be deleted

		paulis.append(pauli)
		expectations.append(float(expectation_string))

	return paulis, expectations