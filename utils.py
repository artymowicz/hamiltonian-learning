import numpy as np
import scipy
import itertools
import functools as ft
import datetime
import os
import time

class SparseTensor:
	def __init__(self,*args, order = False, dtype = None):
		if len(args) == 3:
			shape, indices, values = args
			self.shape = np.asarray(shape, dtype = int)
			self.indices = np.asarray(indices, dtype = int)
			self.values = np.asarray(values, dtype = None)
			assert self.indices.shape == (len(self.values), len(self.shape))
			for i in range(len(shape)):
				assert max(self.indices[:,i])<self.shape[i]
			if order:
				self.order()

		### Initialize from a dense matrix. Not meant to be run on large matrices, used only for testing
		elif len(args) == 1:
			array_like = args[0]
			self.shape = np.array(array_like.shape)
			indices = []
			values = []
			it = np.nditer(array_like, flags=['multi_index'])
			for val in it:
				if val != 0:
					indices.append(it.multi_index)
					values.append(val)
			self.indices = np.array(indices)
			self.values = np.array(values, dtype = dtype)
		else:
			return TypeError(f'SparseTensor.__init__ called with {len(args)} arguments, expected 1 or 3')

	def __str__(self):
		return f'SparseTensor of shape {self.shape} with {len(self.values)} stored elements'

	def __add__(self, other):
		assert np.array_equal(self.shape, other.shape)
		res = SparseTensor(self.shape, np.concatenate((self.indices, other.indices)), np.concatenate((self.values, other.values)))#, order = False)
		res.order()
		res.addUpRedundantEntries()
		return res

	def __sub__(self, other):
		assert np.array_equal(self.shape, other.shape)
		res = SparseTensor(self.shape, np.concatenate((self.indices, other.indices)), np.concatenate((self.values, -other.values)))#, order = False)
		res.order()
		res.addUpRedundantEntries()
		return res

	### checks whether things are of the right type, if indices are out of bounds, and so on
	def consistencyCheck(self):
		if not type(self.shape) == np.ndarray:
			raise ValueError(f'self.shape is wrong type: {type(self.shape)} (expected numpy ndarray)')
		if not type(self.indices) == np.ndarray:
			raise ValueError(f'self.indices is wrong type: {type(self.indices)} (expected numpy ndarray)')
		if not type(self.values) == np.ndarray:
			raise ValueError(f'self.values is wrong type: {type(self.values)} (expected numpy ndarray)')
		if not len(self.values.shape) == 1:
			raise ValueError(f'self.values has wrong shape: {self.values.shape} (expected it to be an array with 1 index)')
		if not self.indices.shape == (len(self.values), len(self.shape)):
			raise ValueError(f'self.indices has wrong shape: {self.indices.shape} ' +
				f'(expected self.indices.shape = (len(self.values), len(self.shape)) = {(len(self.values), len(self.shape))})')

		### checking if any indices are out of bounds:
		for i in range(len(self.shape)):
			indices_out_of_bounds = self.indices[:,i] >= self.shape[i]
			if any(indices_out_of_bounds):
				j = list(indices_out_of_bounds).index(True)
				raise ValueError(f'Index out of bounds. shape is {self.shape}.'
					+ f' The following entry has index {i} out of bounds: index = {self.indices[j]} value = {self.values[j]}')

		### checking if indices are sorted and if duplicates exist
		indices_sorted = True
		no_duplicate_indices = True

		if len(self.values)>1:
			indices_tuples_set = {tuple(self.indices[0])}
			previous_index = self.indices[0]
			for i in range(1,len(self.values)):
				if tuple(self.indices[i]) in indices_tuples_set:
					no_duplicate_indices = False
				else:
					indices_tuples_set.add(tuple(self.indices[i]))
				if tuple(self.indices[i]) <= tuple(previous_index):
					indices_sorted = False
				previous_index = self.indices[i]

		if not indices_sorted:
			print('warning: indices not sorted')
		
		if not no_duplicate_indices:
			raise RuntimeError('duplicate indices found')

	def complexToReal(self,axes):
		assert len(axes) == 2
		(m,n) = self.shape[axes]
		new_shape = self.shape
		new_shape[axes[0]] = 2*self.shape[axes[0]]
		new_shape[axes[1]] = 2*self.shape[axes[1]]

		#upper left block
		new_indices = self.indices
		new_values = np.real(self.values)

		#lower right block
		offset = np.zeros(len(self.shape))
		offset[axes[0]] = m
		offset[axes[1]] = n
		new_indices = np.vstack((new_indices,self.indices + offset))
		new_values = np.concatenate((new_values, np.real(self.values)))

		#lower left block
		offset = np.zeros(len(self.shape))
		offset[axes[0]] = m
		new_indices = np.vstack((new_indices,self.indices + offset))
		new_values = np.concatenate((new_values,np.imag(self.values)))

		#upper right block
		offset = np.zeros(len(self.shape))
		offset[axes[1]] = n
		new_indices = np.vstack((new_indices,self.indices + offset))
		new_values = np.concatenate((new_values,-np.imag(self.values)))

		return SparseTensor(new_shape, new_indices, new_values, order = True, dtype = float)

	### assumes indices are sorted lexicographically
	def addUpRedundantEntries(self):
		if len(self.indices) == 0:
			return
		out_indices = [self.indices[0]]
		out_values = []
		current_total = 0.
		for i in range(len(self.values)):
			if tuple(self.indices[i]) == tuple(out_indices[-1]):
				current_total += self.values[i]
			else:
				out_values.append(current_total)
				current_total = 0.
				out_indices.append(self.indices[i])
				current_total += self.values[i]
		out_values.append(current_total)
		assert len(out_indices) == len(out_values)
		self.indices = np.array(out_indices)
		self.values = np.array(out_values)

	def isEqual(self,other):
		return np.array_equal(self.shape, other.shape) and np.array_equal(self.indices, other.indices) and np.array_equal(self.values, other.values)

	### orders entries by lexicographical order of index
	def order(self, little_endian = False):
		if little_endian:
			nonzeros_indices_sorted = sorted(range(len(self.values)), key = lambda i: list(self.indices[i]).reverse())
		else:
			nonzeros_indices_sorted = sorted(range(len(self.values)), key = lambda i: list(self.indices[i]))
		self.indices = self.indices[nonzeros_indices_sorted]
		self.values = self.values[nonzeros_indices_sorted]

	def conjugate(self):
		return SparseTensor(self.shape,self.indices, np.conjugate(self.values), order = False)

	def copy(self):
		return SparseTensor(self.shape, self.indices, self.values, order = False)

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
		out = np.zeros(self.shape, dtype = self.values.dtype)
		for i in range(len(self.values)):
			out[tuple(self.indices[i])] += self.values[i]
		return out

	def transpose(self, permutation):
		assert len(permutation) == len(self.shape)
		assert sorted(permutation) == list(range(len(self.shape)))

		new_shape = self.shape[permutation]
		new_indices = self.indices[:,permutation]

		return SparseTensor(new_shape, new_indices, self.values, order = True)

	### vectorizes given axes in Fortran order. The new axis is by default the leftmost one
	### WARNING: result is not ordered if called with order = False
	def vectorize(self, axes, order = True):
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

		return SparseTensor(shape_new, indices_new, self.values, order = order)

	### vectorizes a pair of axes, keeping only the lower triangular entries, ie, 
	### those where vectorized indices are in (strictly, if strict = True) decreasing order
	### mosek the the scale_off_diagonals keyword is for mosek's vectorization map
	### WARNING: result is not ordered if called with order = False
	def vectorizeLowerTriangular(self, axes, strict = True, order = True, scale_off_diagonals = None):
		assert len(axes) == 2
		assert self.shape[axes[0]] == self.shape[axes[1]]
		n = self.shape[axes[0]]
		axes_complement = [i for i in range(len(self.shape)) if i not in axes]

		## define the mapping from unvectorized to vectorized indices
		if strict:
			second_index_constants = [0]
			for i in range(n-2):
				second_index_constants.append(second_index_constants[-1] + n-i-1)
			def indexVectorization(index_in):
				(i,j) = index_in[axes]
				if i > j:
					first_index_out = second_index_constants[j] + i-j-1
				else:
					first_index_out = -1
				return np.concatenate(([first_index_out], index_in[axes_complement]))
		else:
			second_index_constants = [0]
			for i in range(n-1):
				second_index_constants.append(second_index_constants[-1] + n-i)
			def indexVectorization(index_in):
				(i,j) = index_in[axes]
				if i >= j:
					first_index_out = second_index_constants[j] + i-j
				else:
					first_index_out = -1
				return np.concatenate(([first_index_out], index_in[axes_complement]))

		### vectorize each index, removing those that are not in the lower triangular part
		indices_new = []
		values_new = []

		if scale_off_diagonals is not None:
			for i in range(len(self.values)):
				index = self.indices[i]
				new_index = indexVectorization(index)
				if new_index[0] >= 0:
					if index[0]>index[1]:
						indices_new.append(new_index)
						values_new.append(self.values[i]*scale_off_diagonals)
					else:
						indices_new.append(new_index)
						values_new.append(self.values[i])
		else:
			for i in range(len(self.values)):
				index = self.indices[i]
				new_index = indexVectorization(index)
				if new_index[0] >= 0:
					indices_new.append(new_index)
					values_new.append(self.values[i])

		## set the new shape
		if strict:
			vec_axis_shape = n*(n-1)//2
		else:
			vec_axis_shape = n*(n+1)//2

		if len(axes_complement) > 0:
			shape_new = np.concatenate(([vec_axis_shape], self.shape[axes_complement]))
		else:
			shape_new = np.array([vec_axis_shape])

		if len(self.shape) == 1:
			self.indices = np.reshape(indices_new, (len(indices_new),1))

		return SparseTensor(shape_new, indices_new, values_new, order = order)
		
	### devectorizes axis given by vector_axis. Expanded axes become the leftmost ones
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
		
		return SparseTensor(self.shape, indices_new, self.values, order = True)

	### Contracts with a dense vector v along the last axis. For instance if A is order 3, returns sum_k A_ijk v_k
	### If skip_ordering = True, assumes that indices are lexicographically (ie. big-endian) ordered
	def contractRight(self,v, order = True):
		if self.shape[-1] != len(v):
			raise ValueError(f'rightmost tensor dimension {self.shape[-1]} does not match vector dimension {len(v)}')

		if order:
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

	### Contracts with a dense vector v along the first axis. For instance if A is order 3, returns sum_i A_ijk v_i
	### If skip_ordering = True, assumes that indices are REVERSE-lexicographically (ie. little-endian) ordered
	def contractLeft(self,v, order = True):
		if self.shape[0] != len(v):
			raise ValueError(f'leftmost tensor dimension {self.shape[0]} does not match vector dimension {len(v)}')

		assert self.shape[-1] == len(v)
		if order:
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

### matrix operations

### same as SparseTensor.vectorizeLowerTriangular but for dense matrices
def vectorizeLowerTriangular(a, strict = True, scale_off_diagonals = None, printing = False):
	assert len(a.shape) == 2
	assert a.shape[0] == a.shape[1]
	n = a.shape[0]

	if scale_off_diagonals is not None:
		diagonal = np.diag(a)
		off_diagonal = a - np.diag(diagonal)
		b = scale_off_diagonals*off_diagonal + np.diag(diagonal)
	else:
		b=a

	if strict:
		k = 1
	else:
		k = 0

	return b.T[np.triu_indices(n, k = k)]

def complexToReal(M):
	assert len(M.shape) == 2
	reM = np.real(M)
	imM = np.imag(M)
	return np.block([[reM, -imM],[imM, reM]])

def realToComplex(M, sanity_check = False):
	assert len(M.shape) == 2
	(m,n) = M.shape
	assert m%2 == 0
	assert n%2 == 0

	if sanity_check:
		assert np.array_equal(M[:m//2,:n//2], M[m//2:, n//2:]) #real part
		assert np.array_equal(M[m//2:,:n//2], -M[:m//2, :n//2]) #imag part

	return M[:m//2,:n//2] + 1j*M[m//2:,:n//2]

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

def weight(pauli_string):
	return sum([1 for c in pauli_string if c != 'I'])

### given a list of unique strings, returns a dict that takes a string and returns the index of that string
def invertStringList(l):
	d = {}
	for i in range(len(l)):
		d[l[i]]= i
	return d

### returns compressed representation of a pauli.
### first number is the Hamming weight
### then, for a single-site pauli factor, we have the letter X,Y,or Z and the index of the corresponding site.
### For instance IYIIIXI is 2 Y 1 X 5 and IIIIZZZ is 3 Z 4 Z 5 Z 6
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

### returns decompressed representation of a pauli. For instance, with n=6, 2 Y 2 X 5 is converted to IIYIIX 
def decompressPauli(pauli_compressed,n):
	pauli_list_out = ['I']*n
	pauli_compressed_list = pauli_compressed.split(' ')
	k = int(pauli_compressed_list[0])
	for i in range(k):
		pauli_char = pauli_compressed_list[1+2*i]
		index = int(pauli_compressed_list[2+2*i])
		pauli_list_out[index] = pauli_char
	return ''.join(pauli_list_out)

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

def buildThreeBodyTerms(onebody_operators, hamiltonian_terms):
	n = len(onebody_operators[0])
	r = len(onebody_operators)
	s = len(hamiltonian_terms)

	onebody_operators = np.asarray(onebody_operators)
	hamiltonian_terms = np.asarray(hamiltonian_terms)

	#noncommuting[i] is a numpy array containing all the indices of onebody operators that don't commute with hamiltonian_terms[i]
	noncommuting = [None]*s
	for i in range(s):
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
	for j in range(s):
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

	return np.hstack((x,y))

### builds a rank-4 tensor C_ijkl such that b_i[h_j,b_k] = sum_l C_ijkl c_l,
### where b_i, b_k are single-body operators, h_j is a hamiltonian term and c_l are three-body operators
def buildTripleProductTensor(onebody_operators, hamiltonian_terms, threebody_operators):
	n = len(onebody_operators[0])
	r = len(onebody_operators)
	h = len(hamiltonian_terms)

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

	### phase_table[4*x+y] is the phase (represented as an integer mod 4) of sigma_x*sigma_y, where x and y are integers between 0 and 3 representing two paulis
	phase_table = np.zeros(16, dtype = 'uint8')
	phase_table[[6,11,13]] = 1
	phase_table[[7,9,14]] = 3

	### compute the commutators [a_j, a_k] and put them into an array whose rows are [j,k, (phase of [a_j,a_k]), pauli of [a_j,a_k] as a list of ints]
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


	shape = np.asarray([r,h,r,len(threebody_operators)])
	indices = np.hstack((combined_indices, last_index))

	out = SparseTensor(shape,indices,total_phases)

	return out

def tprint(s):
	current_time_string = time.strftime("%H:%M:%S", time.localtime())
	print(f'{current_time_string}: {s}')

##### LOADING AND SAVING

### creates a directory to save the results of the run
def createSaveDirectory():
	now = datetime.datetime.now()
	dt_string = now.strftime("%Y_%m_%d-%H_%M_%S")
	if not os.path.exists('./runs/'):
		os.mkdir('./runs/')
	dirname = f'./runs/{dt_string}'
	if not os.path.exists(dirname):
		os.mkdir(dirname)
		return dirname
	else:
		n = 1
		while n < 10:
			dirname_extended = dirname + ' ' + str(n)
			if not os.path.exists(dirname_extended):
				os.mkdir(dirname_extended)
				return dirname_extended
			n+= 1
		raise RuntimeError("tried to make too many save directories with the same name")

