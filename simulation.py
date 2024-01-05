import numpy as np
import scipy
import functools as ft
import argparse
import utils
import itertools
import pickle
import time
import h5py
import yaml
import tenpy
import os
from tqdm import tqdm
from multiprocessing import Pool
import functools

from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO

#parameter used to determine n_chunks in DFScomputeParallel. 
#Generally want it larger than 10, and CHUNKS_PER_THREAD * n_threads << total number of threebody operators
CHUNKS_PER_THREAD = 16

#TODO: replace with numpy native sort
def sortNumpyArray(a, key):
	b = list(a)
	b.sort(key = key)
	return np.asarray(b)

class Hamiltonian:
	def __init__(self, *args):
		assert len(args) == 3
		if type(args[1]) == str:
			assert type(args[0]) == int
			assert type(args[1]) == str
			assert type(args[2]) == dict
			self.n = args[1]
			self.loadHamiltonian(*args)
			self.sort()
		else:
			self.n = args[0]
			self.terms = args[1]
			self.coefficients = args[2]
			self.symmetries = None
			self.sort()

	def __eq__(self, other):
		if self.n != other.n:
			return False

		if self.terms != other.terms:
			return False

		return np.array_equal(self.coefficients, other.coefficients)

	def addTerms(self, new_terms, new_coefficients, skip_sort = False):
		for p in new_terms:
			if p in self.terms:
				raise ValueError(f'term {p} already exists')
		self.terms = np.concatenate((self.terms, new_terms))
		self.coefficients = np.concatenate((self.coefficients, new_coefficients))
		if not skip_sort:
			self.sort()

	def sort(self):
		d = dict(zip(self.terms,self.coefficients))
		self.terms = list(self.terms)
		#self.terms.sort(key = utils.compressPauliToList)
		self.terms = sortNumpyArray(self.terms, key = utils.compressPauliToList)
		self.coefficients = np.asarray([d[p] for p in self.terms])

	def print(self, threshold = 1e-8):
		r = max([utils.weight(p) for p in self.terms])
		line = 'term' + ' '*(4*r-1) +': coefficient'
		print(line)
		print('-'*len(line))
		for i in range(len(self.terms)):
			if np.abs(self.coefficients[i]) > threshold:
				print(f'{utils.compressPauli(self.terms[i])}' +
					' '*(4*r - len(utils.compressPauli(self.terms[i])) + 2) +
					f' :  {self.coefficients[i]:+.6f}')

	def normalizedCoeffs(self,expectations_dict):
		assert self.terms[0] == 'I'*self.n
		out = self.coefficients
		expectations = [expectations_dict[p] for p in self.terms]
		out[0] = -out[1:]@expectations[1:]
		out = out/out[0]
		return out

	def restricted(self,region):
		coefficients_dict = dict(zip(self.terms,self.coefficients))
		terms_restricted = utils.restrictOperators(self.n,self.terms,region)
		coefficients_restricted = np.asarray([coefficients_dict[utils.embedPauli(self.n, p, region)] for p in terms_restricted])
		
		return Hamiltonian(len(region), terms_restricted, coefficients_restricted)

	def normalizeCoeffs(self,expectations_dict):
		assert self.terms[0] == 'I'*self.n
		out = self.coefficients
		expectations = [expectations_dict[p] for p in self.terms]
		out[0] = -out[1:]@expectations[1:]
		normalization = out[0]
		out = out/normalization
		self.coefficients = out
		self.normalization = normalization
		return normalization

	def getNormalization(self):
		if 'normalization' in vars(self):
			return self.normalization
		else:
			return 1

	#MAY REFUSE TO WORK IF periodic = True AND self.terms CONTAINS TERMS OF RANGE MORE THAN n/2 TODO:FIX
	def makeTranslationInvariant(self, periodic):
		new_terms = self.terms
		new_terms_set = set(self.terms)
		new_coeffs_dict = dict(zip(self.terms, self.coefficients))

		for i in range(len(self.terms)):
			for pauli_translated in translates(self.n, self.terms[i], periodic):
				if pauli_translated in new_terms_set:
					raise ValueError('encountered duplicate when adding translates')
				new_terms.append(pauli_translated)
				new_terms_set.add(pauli_translated)
				new_coeffs_dict[pauli_translated] = self.coefficients[i]

		self.terms = new_terms
		self.coefficients = [new_coeffs_dict[p] for p in new_terms]

		self.sort()

	#couplings is a dict
	def loadHamiltonian(self,n,filename, couplings):
		if filename[-4:] == '.txt':
			self.loadHamiltonianFromTextFile(filename)
			return

		with open(filename, 'r') as f:
			#d = yaml.safe_load(f)
			d = yaml.load(f, Loader=yaml.Loader)

		self.n = n
		if n in d:
			assert d['n'] == self.n

		term_coefficient_pairs = d['term_coefficient_pairs']
		
		self.terms =  [utils.decompressPauli(x[0],self.n) for x in term_coefficient_pairs]
		self.coefficients = [x[1] for x in term_coefficient_pairs]

		for i in range(len(self.coefficients)):
			if type(self.coefficients[i]) == str:
				sign = 1
				if self.coefficients[i][0]=='-':
					sign = -1
					self.coefficients[i]=self.coefficients[i][1:]
				elif self.coefficients[i][0]=='+':
					self.coefficients[i]=coefficients[i][1:]
				self.coefficients[i] = sign*couplings[self.coefficients[i]]

		self.symmetries = None

		if 'translation_invariant' in d.keys():
			if d['translation_invariant']:
				self.makeTranslationInvariant(d['periodic'])

		self.sort()

	def saveToYAML(self, filename):
		if filename[-4:] != '.yml':
			filename = filename + '.yml'
		terms_compressed = [utils.compressPauli(p) for p in self.terms]
		term_coefficient_pairs = [list(x) for x in zip(terms_compressed, self.coefficients.tolist())]
		d= dict(n = self.n, term_coefficient_pairs = term_coefficient_pairs)#TODO:implement translation-invariant
		with open(filename, 'w') as f:
			yaml.dump(d,f)

	#magnitude can be a single float or a np array of length disorder_terms
	def addDisorder(self, magnitude, disorder_terms = None, distribution = 'normal'):
		if disorder_terms is None:
			disorder_terms = self.terms

		if distribution == 'normal':
			disorder_coefficients = magnitude*np.random.normal(size = (len(self.terms)))
		else:
			raise ValueError

		terms_indices_dict = utils.invertStringList(self.terms)
		disorder_terms_so_far = set()
		for disorder_term, disorder_coeff in zip(disorder_terms,disorder_coefficients):
			if disorder_term in disorder_terms_so_far:
				raise ValueError(f'found duplicate in disorder terms: {disorder_term}')
			if disorder_term in terms_indices_dict:
				j = terms_indices_dict[disorder_term]
				self.coefficients[j] += disorder_coeff
			else:
				self.terms.append(disorder_term)
				self.coefficients.append(disorder_coeff)

		self.sort()

	#FOR BACKWARDS COMPATIBILITY. in the future switch to YAML only
	def loadHamiltonianFromTextFile(self, filename):
		identity = lambda x : x 
		#keys are expected parameters and values are the function used to parse the corresponding value
		param_conversions = {
							'n' : int,
							'k' : int,				
							'translation_invariant' : utils.strToBool,
							'periodic' : utils.strToBool,
							'locality' : identity
							}

		print(f'loading Hamiltonian from {filename}')
		with open(filename,'r') as f:
			lines = f.readlines()
		l = len(lines)
		raw_params_dict = stringToDict(lines[0][:-1])#line ends in \n so this needs to be deleted
		
		params_dict = dict([(key, param_conversions[key](raw_params_dict[key])) for key in raw_params_dict.keys()])
		n = params_dict['n']

		if params_dict['locality'] == 'short_range':
			terms = utils.buildKLocalPaulis1D(n,params_dict['k'], params_dict['periodic'])
		elif params_dict['locality'] == 'long_range':
			raise ValueError('havent implemented long range interactions')
			### add another parameter for range
		else:
			raise ValueError('unrecognized value ' + params_dict['locality'] + ' for parameter locality')

		coefficients = np.zeros(len(terms))

		##TODO: replace with reading a yaml file 
		if len(lines[1].split(',')) == 3:
			assert lines[1].split(',')[0] == 'pauli'
			assert lines[1].split(',')[2] == 'pauli_compressed\n'
			assert lines[1].split(',')[2] == 'coefficient\n'

			for line in lines[2:]:
				line_split = line.split(',')
				pauli = line_split[0]
				coefficient_string = line_split[2]
				if coefficient_string[-1] == '\n':
					coefficient_string = coefficient_string[:-1]

				if params_dict['translation_invariant']:
					for pauli_translated in translates(pauli):
						if pauli_translated not in terms:
							raise ValueError('Encountered unexpected term ' + pauli_translated + ' when loading Hamiltonian')
						coefficients[terms.index(pauli_translated)] = float(coefficient_string)
				else:
					if pauli not in terms:
						raise ValueError('Encountered unexpected term ' + pauli + ' when loading Hamiltonian')
					coefficients[terms.index(pauli)] = float(coefficient_string)

		if len(lines[1].split(',')) == 2:
			assert lines[1].split(',')[0] == 'pauli'
			assert lines[1].split(',')[1] == 'coefficient\n'

			for line in lines[2:]:
				line_split = line.split(',')
				pauli = utils.decompressPauli(line_split[0],n)
				coefficient_string = line_split[1]
				if coefficient_string[-1] == '\n':
					coefficient_string = coefficient_string[:-1]
				if params_dict['translation_invariant']:
					for pauli_translated in translates(n, pauli, params_dict['periodic']):
						if pauli_translated not in terms:
							raise ValueError('Encountered unexpected term ' + pauli_translated + ' when loading Hamiltonian')
						coefficients[terms.index(pauli_translated)] = float(coefficient_string)
				else:
					if pauli not in terms:
						raise ValueError('Encountered unexpected term ' + pauli + ' when loading Hamiltonian')
					coefficients[terms.index(pauli)] = float(coefficient_string)

			#sort the paulis to make it look nicer
			#sorted_pairs = sorted(zip(terms,coefficients), key = lambda x : utils.compressPauliToList(x[0]))
			#terms = [x[0] for x in sorted_pairs]
			#coefficients = [x[1] for x in sorted_pairs]

		symmetries = [] ### TODO: implement symmetries

		self.n, self.terms, self.coefficients, self.symmetries = n,terms,coefficients, symmetries

### tenpy model that will be used to compute thermal & ground states
class generalSpinHalfModel(CouplingMPOModel):
	default_lattice = Chain
	force_default_lattice = True

	def init_sites(self, model_params):
		conserve = model_params.get('conserve', None)
		assert conserve != 'Sz'
		if conserve == 'best':
			conserve = 'parity'
			self.logger.info("%s: set conserve to %s", self.name, conserve)
		sort_charge = model_params.get('sort_charge', None)
		site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
		return site

	def init_terms(self, model_params):
		ham = model_params.get('ham', None)

		if ham is None:
			raise ValueError('generalSpinHalfModel constructor needs to be given hamiltonian terms and strengths')

		hamiltonian_terms, coefficients = ham
		for term,coefficient in zip(hamiltonian_terms, coefficients):
			if coefficient != 0:
				if len(term) == 1:
					j = term[0][1]
					op = term[0][0]
					self.add_onsite_term(coefficient, j, op)
				elif len(term) == 2:
					j,k = term[0][1], term[1][1]
					op_j, op_k = term[0][0], term[1][0]
					self.add_coupling_term(coefficient, j, k, op_j, op_k)
				else:
					ijkl = [x[1] for x in term]
					ops_ijkl = [x[0] for x in term]
					self.add_multi_coupling_term(coefficient, ijkl, ops_ijkl, 'Id')


class EquilibriumState:
	def __init__(self, n, H, H_name, beta):
		self.n = n
		self.H = H
		self.H_name = H_name
		self.beta = beta
		self.psi = None
		self.metrics = {}

	def computeEquilibriumState(self, simulator_params):
		method, skip_intermediate_betas = simulator_params['simulator_method'], simulator_params['simulator_skip_intermediate_betas']

		### compute the state
		if method == 'tenpy':
			t1 = time.time()
			if self.beta == np.inf:
				utils.tprint('computing state using DMRG')
				self.psi = computeGroundstateDMRG(self.H, simulator_params)
				t2 = time.time()
				self.metrics['rho_computation_time_by_DMRG'] = t2-t1
			else:
				utils.tprint('computing state using purification')
				self.psi = computeThermalStateByPurification(self.H, self.beta, simulator_params)
				t2 = time.time()
				self.metrics['rho_computation_time_by_MPO'] = t2-t1

	def computeExpectations(self, operators, params):
		if self.psi == None:
			self.computeEquilibriumState(params)
		if params['simulator_method'] == 'tenpy':
			n_chunks = params['n_threads']*CHUNKS_PER_THREAD
			t1 = time.time()
			if self.beta == np.inf:
				expectations, tenpy_calls = DFScomputeParallel(self.n, self.psi, operators, 
					'pure', params['n_threads'], n_chunks, printing = params['printing'], naive = params['naive_compute'])
			else:
				expectations, tenpy_calls = DFScomputeParallel(self.n, self.psi, operators, 
					'mixed', params['n_threads'], n_chunks, printing = params['printing'], naive = params['naive_compute'])
			t2 = time.time()
			self.metrics['tenpy_calls'] = tenpy_calls
			self.metrics['expectations_computation_time'] = t2-t1
		elif params['simulator_method'] == 'ED':
			expectations = computeExpectationsED(operators)
		else:
			raise ValueError("only valid input so far is 'tenpy'")
		return expectations

	def getExpectations(self, operators, params):
		if params['no_cache']:
			return self.computeExpectations(operators, params)

		if not os.path.exists('./caches/'):
			os.mkdir('./caches/')
		filename = f"./caches/{self.H_name}_exp_cache.hdf5"

		try:
			exp_file = h5py.File(filename, 'r+')
			utils.tprint(f'checking that cached expectations hamiltonian agrees with given one')
			a = np.array_equal(np.char.decode(exp_file['/hamiltonian/terms']), self.H.terms)
			b = np.array_equal(exp_file['/hamiltonian/coeffs'], self.H.coefficients)
			if not (a and b):
				#print(exp_file['/hamiltonian/terms'][:])
				#print(hamiltonian.terms)
				raise ValueError(f'cached expectations hamiltonian does not agree with given one. To recompute, remove {filename} from caches directory and rerun')
			
		except FileNotFoundError:
			utils.tprint(f'expectations cache at {filename} not found. Creating one')
			exp_file = h5py.File(filename, 'w')
			exp_file['/hamiltonian/terms'] = np.char.encode(self.H.terms)
			exp_file['/hamiltonian/coeffs'] = self.H.coefficients
			exp_file.create_group('/expectations')

		#expectations for a given inverse temperature beta are saved in /expectations/{beta rounded to 5 digits after decimal point} or /expectations/inf for groundstate
		beta_str = f'{self.beta:.5f}'

		if beta_str not in exp_file['expectations'].keys():
			utils.tprint(f'cached expectations for beta = {beta_str} not found')
		elif params['overwrite_cache']:
			current_time_string = time.strftime("%H:%M:%S", time.localtime())
			utils.tprint(f'overwriting expectations cache')
			del exp_file['expectations/' + beta_str]
		else:
			utils.tprint(f'cached expectations for beta = {beta_str} found. decoding strings')
			cached_operators = np.char.decode(exp_file[f'expectations/{beta_str}/operators'])
			if params['skip_checks'] is False:
				utils.tprint(f'checking that all required expectations are cached')

				##TODO: this can be done in numpy for speed
				all_there = True
				operators_set = set(cached_operators)
				for t in operators:
					if t not in operators_set:
						print(f'operator {t} not found in cache')
						all_there = False
						break

				if all_there:
					expectations_dict = dict(zip(cached_operators,exp_file[f'expectations/{beta_str}/exp_values']))
					utils.tprint(f'all required expectations found in cache')
					return [expectations_dict[p] for p in operators]
			else:
				expectations_dict = dict(zip(cached_operators,exp_file[f'expectations/{beta_str}/exp_values']))
				return [expectations_dict[p] for p in operators]

		utils.tprint('computing expectation values using method '+ params['simulator_method'])
		#expectations_dict, metrics = computeRequiredExpectations(hamiltonian, beta, simulator_params)
		expectations = self.computeExpectations(operators, params)
		expectations_dict = dict(zip(operators, expectations))

		if beta_str in exp_file['expectations'].keys():
			utils.tprint('checking consistency of newly computed expectation values with old ones')
			existing_operators = np.char.decode(exp_file[f'expectations/{beta_str}/operators'])
			existing_expectations = exp_file[f'expectations/{beta_str}/exp_values']
			existing_expectations_dict = dict(zip(existing_operators, existing_expectations))

			### TODO: test this

			if not np.allclose([existing_expectations_dict[o]-expectations_dict[o] for o in operators if o in existing_expectations_dict],0, atol=1e-08):
				tmp_filename = './caches/tmp.hdf5'
				with h5py.File(tmp_filename, 'w') as tmp_file:
					tmp_file['hamiltonian/terms'] = np.char.encode(self.H.terms)
					tmp_file['hamiltonian/coeffs'] = self.H.coeffs
					tmp_file[f'expectations/{beta_str}/operators'] = np.char.encode(existing_operators)
					tmp_file[f'expectations/{beta_str}/exp_values'] = existing_expectations
					print(f'warning: some expectations were not consistent with previously computed expectations'
					+ 'Saving previous expectations in {tmp_filename} and overwriting')

			del exp_file[f'expectations/{beta_str}/operators']
			del exp_file[f'expectations/{beta_str}/exp_values']

		utils.tprint('saving expectation values just computed')
		#TODO: this can be done purely in numpy to be faster
		data_sorted = sorted(zip(expectations_dict.keys(),expectations_dict.values()), key = lambda x: utils.compressPauliToList(x[0]))
		exp_file[f'expectations/{beta_str}/operators'] = np.char.encode([x[0] for x in data_sorted])
		exp_file[f'expectations/{beta_str}/exp_values'] = [x[1] for x in data_sorted]

		exp_file.close()
		return [expectations_dict[p] for p in operators]

	def computeExpectationsED(self, operators):
		load = True
		n = self.H.n

		utils.tprint('building many-body hamiltonian matrix')
		### build many-body hamiltonian matrix
		H_mb = np.zeros((2**n,2**n), dtype = complex)
		for i in range(len(self.H.terms)):
			if len(self.H.terms[i]) != n:
				raise ValueError(f'{self.H.terms[i]} is not the correct length, expected {n}')
			H_mb += self.H.coefficients[i]*utils.pauliMatrix(self.H.terms[i])

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
		if self.beta == np.inf:
			_, eigvecs = scipy.linalg.eigh(H_mb)
			psi = eigvecs[:,0]
			rho = np.conjugate(psi.reshape((2**n,1)))@psi.reshape((1,2**n))
		else:
			rho = scipy.linalg.expm(-self.beta*H_mb)
			rho = rho/np.trace(rho)

		print(f'rho.shape = {rho.shape}')
		out = []
		for p in tqdm(operators):
			out.append(utils.computeExpectation(p,rho))
		return out

###DEPRECATED
def requiredExpectations(hamiltonian, onebody_operators, hamiltonian_terms, beta, simulator_params):
	compute = True

	if simulator_params['exp_cache'] is None:
		#return computeRequiredExpectations(hamiltonian, beta, simulator_params)
		return computeRequiredExpectations2(hamiltonian, onebody_operators, hamiltonian_terms, beta, simulator_params)
	filename = simulator_params['exp_cache']
	try:
		'''
		with open(filename, 'rb') as f:
			expectations_dict = pickle.load(f)
			### TODO: check if cached expectations match hamiltonian, method, and beta
			print(f'expectations cache found at {filename}')
		'''
		exp_file = h5py.File(filename, 'r+')
		utils.tprint(f'checking that cached expectations hamiltonian agrees with given one')
		a = np.array_equal(np.char.decode(exp_file['/hamiltonian/terms']), hamiltonian.terms)
		b = np.array_equal(exp_file['/hamiltonian/coeffs'], hamiltonian.coefficients)
		if not (a and b):
			print(exp_file['/hamiltonian/terms'][:])
			print(hamiltonian.terms)
			raise ValueError(f'cached expectations hamiltonian does not agree with given one. To recompute, remove {filename} from caches directory and rerun')
		
	except FileNotFoundError:
		utils.tprint(f'expectations cache at {filename} not found. Creating one')
		exp_file = h5py.File(filename, 'w')
		exp_file['/hamiltonian/terms'] = np.char.encode(hamiltonian.terms)
		exp_file['/hamiltonian/coeffs'] = hamiltonian.coefficients
		exp_file.create_group('/expectations')

	#expectations for a given inverse temperature beta are saved in /expectations/{beta rounded to 5 digits after decimal point} or /expectations/inf for groundstate
	beta_str = f'{beta:.5f}'

	if beta_str not in exp_file['expectations'].keys():
		utils.tprint(f'cached expectations for beta = {beta_str} not found')
	elif simulator_params['overwrite_cache']:
		current_time_string = time.strftime("%H:%M:%S", time.localtime())
		utils.tprint(f'overwriting expectations cache')
		del exp_file['expectations/' + beta_str]
	else:
		utils.tprint(f'cached expectations for beta = {beta_str} found. decoding strings')
		operators = np.char.decode(exp_file[f'expectations/{beta_str}/operators'])
		if simulator_params['skip_checks'] is False:
			utils.tprint(f'computing required operators')
			required_operators = generateRequiredOperators(hamiltonian.n, hamiltonian.terms, onebody_operators)
			utils.tprint(f'checking that all required expectations are cached')

			##TODO: this can be done in numpy for speed
			all_there = True
			operators_set = set(operators)
			for t in required_operators:
				if t not in operators_set:
					print(f'operator {t} not found in cache')
					all_there = False
					break

			if all_there:
				expectations_dict = dict(zip(operators,exp_file[f'expectations/{beta_str}/exp_values']))
				metrics = {}
				return expectations_dict, metrics
		else:
			expectations_dict = dict(zip(operators,exp_file[f'expectations/{beta_str}/exp_values']))
			metrics = {}
			return expectations_dict, metrics

	#print('computing expectation values using method '+ simulator_params['method'])
	#expectations_dict, metrics = computeRequiredExpectations(hamiltonian, beta, simulator_params)
	expectations_dict, metrics = computeRequiredExpectations2(hamiltonian, onebody_operators, hamiltonian_terms, beta, simulator_params)

	if beta_str in exp_file['expectations'].keys():
		utils.tprint('checking consistency of newly computed expectation values with old ones')
		existing_operators = np.char.decode(exp_file[f'expectations/{beta_str}/operators'])
		existing_expectations = exp_file[f'expectations/{beta_str}/exp_values']

		### TODO: test this
		if not numpy.allclose(existing_expectations, [expectations_dict[o] for o in existing_operators], atol=1e-08):
			tmp_filename = './caches/tmp.hdf5'
			with h5py.File(tmp_filename, 'w') as tmp_file:
				tmp_file['hamiltonian/terms'] = np.char.encode(hamiltonian.terms)
				tmp_file['hamiltonian/coeffs'] = hamiltonian.coeffs
				tmp_file[f'expectations/{beta_str}/operators'] = np.char.encode(existing_operators)
				tmp_file[f'expectations/{beta_str}/exp_values'] = existing_expectations
				print(f'warning: some expectations were not consistent with previously computed expectations'
				+ 'Saving previous expectations in {tmp_filename} and overwriting')

		del exp_file[f'expectations/{beta_str}/operators']
		del exp_file[f'expectations/{beta_str}/exp_values']

	utils.tprint('saving expectation values just computed')
	#TODO: this can be done purely in numpy to be faster
	data_sorted = sorted(zip(expectations_dict.keys(),expectations_dict.values()), key = lambda x: utils.compressPauliToList(x[0]))
	exp_file[f'expectations/{beta_str}/operators'] = np.char.encode([x[0] for x in data_sorted])
	exp_file[f'expectations/{beta_str}/exp_values'] = [x[1] for x in data_sorted]

	exp_file.close()
	return expectations_dict, metrics

def DFScomputeOld(n, psi, operators, printing = False):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*n
	tenpy_calls = 0
	for i in range(l):
		if i%(l//100) == 0:
			print(f'{i/l:.0%} done')
		p = operators[i]
		j = utils.firstDifferingIndex(p, previous)
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

	return dict(zip(operators,out)), tenpy_calls

def DFScomputeParallel(n, psi, operators, state_type, n_threads, n_chunks, printing = False, naive = False):
	if printing:
		utils.tprint(f'computing expectation values. n_theads = {n_threads}')
	l = len(operators)
	out = np.zeros(l)
	ind_list = [(l//n_chunks)*k for k in range(n_chunks)]
	ind_list += [l]
	operators_chunks = [operators[ind_list[i]:ind_list[i+1]] for i in range(n_chunks)]
	#f = lambda x : DFScomputeNew(n, psi, x, printing = printing)
	if naive:
		f = functools.partial(DFScomputeSingleThreadNaive, n, psi)
	else:
		if state_type == 'pure':
			f = functools.partial(DFScomputeSingleThreadPure, n, psi)
		elif state_type == 'mixed':
			f = functools.partial(DFScomputeSingleThreadMixed, n, psi)
		else:
			raise ValueError
	with Pool(n_threads) as p:
		results = p.map(f, operators_chunks)
	out = np.concatenate(tuple([x[0] for x in results]))
	return out, sum([x[1] for x in results]) 

def DFScomputeSingleThreadPure(n, psi, operators):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*(n+1)
	L_tensors[0] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vR', 'vR*'])

	tenpy_calls = 0

	#print(f'running DFSComputeNew with n = {n}, l = {l}')

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
		#if i%(l//100) == 0:
		#	utils.tprint(f'{i/l:.0%} done')
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


		y = max(j,last_nontriv)

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

	#print(f'total number of contractions = {tenpy_calls}')
	#print(f'{tenpy_calls/l - 1} nontrivial contractions per evaluation')
	return out, tenpy_calls

def DFScomputeSingleThreadNaive(n,psi,operators):
	return [psi.expectation_value_term(pauliStringToTenpyTerm(n,p)) for p in operators], len(operators)

def DFScomputeSingleThreadMixed(n, psi, operators):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*(n+1)
	L_tensors[0] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vR', 'vR*'])

	tenpy_calls = 0

	#print(f'running DFSComputeNew with n = {n}, l = {l}')

	###compute R_tensors
	R_tensors = [None]*(n+1)
	R_tensors[n] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vL', 'vL*'])
	for k in range(n,0,-1):
		R = R_tensors[k]
		B = psi.get_B(k-1)
		tensor_list = [R,B, B.conj()]
		tensor_names = ['R', 'B','B*']
		leg_contractions = [['B', 'p', 'B*', 'p*'], ['B', 'q', 'B*', 'q*']]
		leg_contractions += [['R', 'vL', 'B', 'vR'], ['R', 'vL*', 'B*', 'vR*']]
		open_legs = [['B', 'vL', 'vL'], ['B*', 'vL*', 'vL*']]
		R_tensors[k-1] = tenpy.algorithms.network_contractor.contract(tensor_list, tensor_names, leg_contractions, open_legs)
		#tenpy_calls += 1

	for i in range(l):
		#if i%(l//100) == 0:
		#	utils.tprint(f'{i/l:.0%} done')
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

		y = max(j,last_nontriv)

		for k in range(j,y+1):
				L = L_tensors[k]
				B = psi.get_B(k)
				onsite_term = psi.sites[k].get_op(my_to_tenpy[p[k]])
				tensor_list = [L, B, B.conj(), onsite_term]
				tensor_names = ['L', 'B','B*', 'O']
				leg_contractions = [['B', 'q', 'B*', 'q*']]
				leg_contractions += [['B', 'p', 'O', 'p']]
				leg_contractions += [['B*', 'p*', 'O', 'p*']]
				leg_contractions += [['L', 'vR', 'B', 'vL'], ['L', 'vR*', 'B*', 'vL*']]
				open_legs = [['B', 'vR', 'vR'], ['B*', 'vR*', 'vR*']]
				L_tensors[k+1] = tenpy.algorithms.network_contractor.contract(tensor_list, tensor_names, leg_contractions, open_legs)
				tenpy_calls += 1

		out[i] = np.real(tenpy.algorithms.network_contractor.contract([L_tensors[y+1], R_tensors[y+1]], ['L', 'R'], [['L','vR','R','vL'], ['L','vR*','R','vL*']]))
		tenpy_calls += 1
		previous = p

	#print(f'total number of contractions = {tenpy_calls}')
	#print(f'{tenpy_calls/l - 1:.1f} nontrivial contractions per evaluation')
	return out, tenpy_calls

#def computeThreeBodyOperators(onebody_operators, hamiltonian_terms):
#	return sorted(utils.buildTripleProductTensor(onebody_operators, hamiltonian_terms)[1])

def computeRequiredExpectations2(hamiltonian, onebody_operators, hamiltonian_terms, beta, simulator_params):
	n = hamiltonian.n
	method, skip_intermediate_betas = simulator_params['method'], simulator_params['skip_intermediate_betas']

	### compute the state
	if method == 'tenpy':
		t1 = time.time()
		if beta == np.inf:
			utils.tprint('computing state using DMRG')
			psi = computeGroundstateDMRG(hamiltonian, simulator_params)
		else:
			utils.tprint('computing state using purification')
			psi = computeThermalStateByPurification(hamiltonian,beta, simulator_params)
		t2 = time.time()

	### compute required operators, sorted 
	threebody_operators = computeThreeBodyOperators(onebody_operators, hamiltonian_terms)
	t3 = time.time()

	### compute the expectations using DFS
	utils.tprint('computing expectations')
	threebody_expectations, tenpy_calls = DFScomputeParallel(n, psi, threebody_operators, 4, printing = True)
	expectations_dict = dict(zip(threebody_operators, threebody_expectations))
	t4 = time.time()

	metrics = {}
	metrics['rho_computation_time_by_MPO'] = t2-t1
	metrics['threebody_operators_generate_time'] = t3-t2
	metrics['expectations_computation_time'] = t4-t3
	metrics['tenpy_calls'] = tenpy_calls
	metrics['simulator_params'] = simulator_params

	return expectations_dict, metrics 

#DEPRECATED
def computeRequiredExpectations(hamiltonian, beta, simulator_params):
	n = hamiltonian.n
	method, skip_intermediate_betas = simulator_params['method'], simulator_params['skip_intermediate_betas']

	if method == 'tenpy':
		t1 = time.time()
		if beta == np.inf:
			utils.tprint('computing state using DMRG')
			psi = computeGroundstateDMRG(hamiltonian, simulator_params)
		else:
			utils.tprint('computing state using purification')
			psi = computeThermalStateByPurification(hamiltonian,beta, simulator_params)
		t2 = time.time()

		#TODO: generalize this. So far works only for k=2 and locality = short_range
		if skip_intermediate_betas:
			utils.tprint('computing expectations')
			#expectations_dict, tenpy_calls = computeCorrelators(n, 3,4, psi ,return_tenpy_calls = True, printing = simulator_params['printing']) 
			expectations_dict, tenpy_calls = compute2by3correlators(n, psi, return_tenpy_calls = True, printing = simulator_params['printing']) 
		else:
			expectations_dicts, metrics = batchCompute2by3correlators(hamiltonian.n, psis)
		t3 = time.time()

	metrics = {}
	metrics['rho_computation_time_by_MPO'] = t2-t1
	metrics['expectations_computation_time_from_MPO'] = t3-t2
	metrics['tenpy_calls'] = tenpy_calls
	metrics['simulator_params'] = simulator_params

	return expectations_dict, metrics

#TODO: make this spit out a numpy str array
#DEPRECATED
def generateRequiredOperators(n, hamiltonian_terms, onebody_terms):
	###TODO: write an actual function
	return build2By3CorrelatorList(n)

pauli_generators = {'X': np.array([[0,1],[1,0]], dtype = complex),
					'Y': np.array([[0,0.-1.j],[0.+1.j,0]]),
					'Z': np.array([[1.,0.],[0,-1.]], dtype=complex),
					'I': np.identity(2, dtype=complex)}

tenpy_paulis = ['Id','Sigmax','Sigmay','Sigmaz']
my_paulis = ['I','X','Y','Z']
tenpy_to_my = dict(zip(tenpy_paulis,my_paulis))
my_to_tenpy = dict(zip(my_paulis,tenpy_paulis))

def pauliStringToTenpyTerm(n,p):
	if p == 'I'*n:
		return [('Id',0)]
	else:
		return [(my_to_tenpy[p[i]],i) for i in range(n) if p[i] != 'I']

def tenpyTermToPauliString(n,term):
	out_list = ['I']*n
	for op in term:
		out_list[op[1]] = tenpy_to_my[op[0]]
	return ''.join(out_list)

def translateTerm(term, i):
	return [(x[0], x[1]+i) for x in term]

'''
def buildTerms(n,k):
	onebody_operators = utils.buildKLocalPaulis1D(n,k, periodic_bc = False)
	_, threebody_paulis = hamiltonian_learning.buildTripleProductTensor(onebody_operators, onebody_operators)
	threebody_paulis.sort(key = utils.compressPauli)
	print(len(threebody_paulis))
	onebody_terms = [pauliStringToTerm(n,p) for p in onebody_paulis]
	threebody_terms = [pauliStringToTerm(n,p) for p in threebody_paulis]
	return onebody_terms, threebody_terms
'''

twosite_terms = [[(a,0), (b,1)] for (a,b) in itertools.product(tenpy_paulis,tenpy_paulis)]
threesite_terms = [[(a,0), (b,1), (c,2)] for (a,b,c) in itertools.product(tenpy_paulis,tenpy_paulis,tenpy_paulis)]

def build2By3CorrelatorList(n):
	out = []
	for x,y in itertools.product(twosite_terms, threesite_terms):
		for i in range(n-1):
			nonoverlapping_j = [j for j in range(n-2) if j<i-2 or j>i+1]
			current_paulis = [tenpyTermToPauliString(n, translateTerm(x,i)+translateTerm(y,j)) for j in nonoverlapping_j]
			for p in current_paulis:
				if p not in out:
					out.append(p)
	return out

#TODO: write. maybe here I can do some multithreading, or something in cython?
#DEPRECATED
def batchCompute2by3correlators(n, psis):
	return

#these are of the form <AB> where A is a 2-local operator, B is a 3-local operator, and A and B don't overlap
#DEPRECATED
def compute2by3correlators(n, psi, return_tenpy_calls = False, printing = True):
	d = {}
	calls_to_tenpy = 0
	total_calls_to_tenpy = 2048*(n-4)
	#### two site term on the left, three-site on the right
	for x,y in itertools.product(twosite_terms, threesite_terms):
		for i in range(n-1):
			nonoverlapping_j = [j for j in range(n-2) if j>i+1]
			if len(nonoverlapping_j)>0:
				current_paulis = [tenpyTermToPauliString(n, translateTerm(x,i)+translateTerm(y,j)) for j in nonoverlapping_j]
				corr = psi.term_correlation_function_right(x, y, i_L=i, j_R=nonoverlapping_j)
				calls_to_tenpy += 1
				if calls_to_tenpy%(np.ceil(total_calls_to_tenpy/100))==0 and printing:
					utils.tprint(f'{calls_to_tenpy/total_calls_to_tenpy:.1%} done')
				for i in range(len(current_paulis)):
					d[current_paulis[i]] = corr[i]

	#### three site term on the left, two-site on the right
	for x,y in itertools.product(threesite_terms, twosite_terms):
		for i in range(n-2):
			nonoverlapping_j = [j for j in range(n-1) if j>i+2]
			if len(nonoverlapping_j)>0:
				current_paulis = [tenpyTermToPauliString(n, translateTerm(x,i)+translateTerm(y,j)) for j in nonoverlapping_j]
				corr = psi.term_correlation_function_right(x, y, i_L=i, j_R=nonoverlapping_j)
				calls_to_tenpy += 1
				if calls_to_tenpy%(np.ceil(total_calls_to_tenpy/100))==0 and printing:
					utils.tprint(f'{calls_to_tenpy/total_calls_to_tenpy:.1%} done')
				for i in range(len(current_paulis)):
					d[current_paulis[i]] = corr[i]
	if return_tenpy_calls:
		return d, calls_to_tenpy
	else:
		return d

def allTerms(k):
	return [[(x[i],i) for i in range(k)] for x in itertools.product(tenpy_paulis, repeat = k)]

def computeCorrelators(n, k,l, psi, return_tenpy_calls = False, printing = True):
	d = {}
	calls_to_tenpy = 0

	if k == l:
		total_calls_to_tenpy = (n-k-l+1)*(4**(k+l))
	else:
		total_calls_to_tenpy = 2*(n-k-l+1)*(4**(k+l))

	k_site_terms = allTerms(k)
	l_site_terms = allTerms(l)

	#### k-site term on the left, l-site term on the right
	for x,y in itertools.product(k_site_terms, l_site_terms):
		for i in range(n-k+1):
			nonoverlapping_j = [j for j in range(n-l+1) if j>i+k-1]
			if len(nonoverlapping_j)>0:
				current_paulis = [tenpyTermToPauliString(n, translateTerm(x,i)+translateTerm(y,j)) for j in nonoverlapping_j]
				corr = psi.term_correlation_function_right(x, y, i_L=i, j_R=nonoverlapping_j)
				calls_to_tenpy += 1
				if calls_to_tenpy%(np.ceil(total_calls_to_tenpy/1000))==0 and printing:
					utils.tprint(f'{calls_to_tenpy/total_calls_to_tenpy:.1%} done')
				for i in range(len(current_paulis)):
					d[current_paulis[i]] = corr[i]
	if k != l:
		#### l-site term on the left, k-site term on the right
		for x,y in itertools.product(l_site_terms, k_site_terms):
			for i in range(n-l+1):
				nonoverlapping_j = [j for j in range(n-k+1) if j>i+l-1]
				if len(nonoverlapping_j)>0:
					current_paulis = [tenpyTermToPauliString(n, translateTerm(x,i)+translateTerm(y,j)) for j in nonoverlapping_j]
					corr = psi.term_correlation_function_right(x, y, i_L=i, j_R=nonoverlapping_j)
					calls_to_tenpy += 1
					if calls_to_tenpy%(np.ceil(total_calls_to_tenpy/100))==0 and printing:
						utils.tprint(f'{calls_to_tenpy/total_calls_to_tenpy:.1%} done')
					for i in range(len(current_paulis)):
						d[current_paulis[i]] = corr[i]

	if return_tenpy_calls:
		return d, calls_to_tenpy
	else:
		return d

#given kth vector in computational basis, compute its image under translation
def rotateBasis(k,n):
	bitstring = format(k, '0{}b'.format(n))#converting to binary
	bitstring_rotated = bitstring[1:]+b[0]
	return int(bitstring_rotated,2)#converting back to int

#sparse matrix corresponding to lattice translation
def generateRotationMatrix(n):
	data = np.ones(2**n)
	rows = [rotateBasis(k,n) for k in range(2**n)]
	cols= range(2**n)
	out = scipy.sparse.coo_array(data,(rows,cols))
	return out

def generatePauliMatrix(pauli_string):
	pauli_list = [scipy.sparse.coo_array(pauli_generators[c]) for c in pauli_string]
	return ft.reduce(scipy.sparse.kron, pauli_list)

def onsitePauliString(i, pauli, N):
	return 'I'*i + pauli + 'I'*(N-i-1)

def twoSitePauliString(i,j, pauli_1, pauli_2, N):
	assert i != j
	if i > j:
		(i,j) = (j,i)
		(pauli_1,pauli_2) = (pauli_2,pauli_1)
	out = 'I'*i + pauli_1 + 'I'*(j-i-1) + pauli_2 + 'I'*(N-j-1)
	return out

def threeSitePauliString(i,j,k, pauli_1, pauli_2, pauli_3, N):
	assert i != j != k
	(i,pauli_1), (j,pauli_2), (k,pauli_3) = sorted(((i,pauli_1), (j,pauli_2), (k,pauli_3)))
	out = 'I'*i + pauli_1 + 'I'*(j-i-1) + pauli_2 + 'I'*(k-j-1) + pauli_3 + 'I'*(N-k-1)
	return out

def exactDiagonalization(n,paulis,coefficients, return_spectrum = False):
	#build Hamiltonian
	H = scipy.sparse.coo_array((2**n,2**n))
	assert len(paulis) == len(coefficients)

	for i in range(len(paulis)):
		if len(paulis[i]) != n:
			raise ValueError(f'{paulis[i]} is not the correct length, expected {n}')
		H += coefficients[i]*generatePauliMatrix(paulis[i])

	if return_spectrum == False:
		if n == 1:
			_, psi = scipy.linalg.eigh(H.toarray())
		else:
			_, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
			#_, psi = scipy.linalg.eigh(H.toarray())
		psi = psi[:,0]
		return psi

	else:
		eigvals, eigvectors = scipy.linalg.eigh(H.toarray())
		psi = eigvectors[:,0]
		return psi, eigvals

def signFromSupport(n,support):
	total = 0
	for i in range(2**n):
		if support[i]:
			pass

#given a state as a full wavefunction, compute expectation of a given operator
def computeExpectationFromWaveFunc(pauli_string, psi, n):
	support_bitstring = utils.supportBitString(pauli_string)
	support_bitstring_int = int(support_bitstring,2)
	U = utils.generateCliffordMatrix(pauli_string)
	weights = np.square(np.absolute(U@psi))
	#signs = np.ones(2**n) - 2*np.bitwise_count(np.bitwise_and(support_bitstring_int,np.arange(2**n)))# for when numpy 2.0 comes out
	signs = np.ones(2**n) - [2*(x.bit_count()%2) for x in np.bitwise_and(support_bitstring_int,np.arange(2**n))]
	out = np.dot(signs, weights)
	return out

#given a (left-justified) pauli returns all its translates
def translates(n, pauli, periodic_bc):
	assert n>0
	assert pauli[0] != 'I'

	pauli_translates = []

	if periodic_bc:
		for i in range(1,n):
			pauli_rotated = pauli[n-i:] + pauli[:n-i]
			if pauli_rotated in pauli_translates:
				print(f'warning: duplicate found when constructing translation-invariant hamiltonian: {pauli_rotated}')
			else:
				pauli_translates.append(pauli_rotated)
	else:
		p_len = max([i for i in range(n) if pauli[i] != 'I'])+1
		for i in range(1,n-p_len+1):
			pauli_rotated = pauli[n-i:] + pauli[:n-i]
			if pauli_rotated in pauli_translates:
				print(f'warning: duplicate found when constructing translation-invariant hamiltonian: {pauli_rotated}')
			else:
				pauli_translates.append(pauli_rotated)

	return pauli_translates

### DEPRECATED
def addTranslates(n, paulis, coefficients, periodic_bc):
	paulis_translates = paulis
	l = len(paulis)
	paulis_translates_set = set(paulis)
	coefficients_translates = coefficients

	#we depend on this assumption
	for i in range(l):
		assert paulis[i][0] != 'I'

	if periodic_bc:
		for i in range(1,n):
			for j in range(l):
				pauli_rotated = paulis[j][i:] + paulis[j][:i] 
				if pauli_rotated in paulis_translates_set:
					print(f'warning: duplicate found when constructing translation-invariant hamiltonian: {pauli_rotated}')
				else:
					paulis_translates_set.add(pauli_rotated)
					paulis_translates.append(pauli_rotated)
					coefficients_translates.append(coefficients[j])
	else:
		for i in range(1,n):
			for j in range(l):
				pauli_rotated = paulis[j][i:] + paulis[j][:i] 
				if pauli_rotated[0]=='I':
					if pauli_rotated in paulis_translates_set:
						print(f'warning: duplicate found when constructing translation-invariant hamiltonian: {pauli_rotated}')
					else:
						paulis_translates_set.add(pauli_rotated)
						paulis_translates.append(pauli_rotated)
						coefficients_translates.append(coefficients[j])

	return paulis_translates, coefficients_translates

def scatterPlot(l, label = None):
	r = range(len(l))
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(r, l , s=2, c='b', marker="s", label = label)
	plt.show()

def saveState(n,psi, periodic, filename, dir = './states/'):
	assert len(psi) == 2**n
	with open(dir+filename, 'w') as f:
		f.write(str(n)+'\n')
		'''
		if periodic:
			f.write('periodic : 1\n')
		else:
			f.write('periodic : 0\n')
		'''
		re_psi = np.real(psi)
		im_psi = np.imag(psi)
		for i in range(2**n):
			f.write(str(re_psi[i]) + ' ' + str(im_psi[i]) + '\n')

def checkTranslationInvariance(n,H, psi, threshold = 1e-10, printing = True):
	R = generateRotationMatrix(n)
	H_rot = R@H@R.T
	psi_rot = R@psi

	H_equality = scipy.sparse.linalg.norm(H-H_rot) < threshold
	psi_equality = np.linalg.norm(psi-psi_rot) < threshold

	if printing:
		print(f'equality check between RHR^* and H: {H_equality}')
		print(f'equality check between Rpsi and psi: {psi_equality})')

	return H_equality, psi_equality

#converts a string of the form "key_1 : val_1, key_2 : val_2" to a dict
def stringToDict(s):
	d = {}
	s_split = s.split(', ')
	for x in s_split:
		x_split = x.split(' : ')
		assert len(x_split) == 2
		d[x_split[0]] = x_split[1]
	return d

identity = lambda x : x 
#keys are expected parameters and values are the function used to parse the corresponding value
param_conversions = {
					'n' : int,
					'k' : int,				
					'translation_invariant' : utils.strToBool,
					'periodic' : utils.strToBool,
					'locality' : identity
					}

def loadHamiltonian(filename):
	ham_filename = './hamiltonians/'+filename + '.txt'
	print(f'loading Hamiltonian from {ham_filename}')
	with open(ham_filename,'r') as f:
		lines = f.readlines()
	l = len(lines)
	raw_params_dict = stringToDict(lines[0][:-1])#line ends in \n so this needs to be deleted
	
	params_dict = dict([(key, param_conversions[key](raw_params_dict[key])) for key in raw_params_dict.keys()])
	n = params_dict['n']

	if len(lines[1].split(',')) == 3:
		assert lines[1].split(',')[0] == 'pauli'
		assert lines[1].split(',')[2] == 'pauli_compressed\n'
		assert lines[1].split(',')[2] == 'coefficient\n'

		paulis = []
		coefficients = []
		for line in lines[2:]:
			line_split = line.split(',')
			pauli = line_split[0]
			coefficient_string = line_split[2]
			if coefficient_string[-1] == '\n':
				coefficient_string = coefficient_string[:-1]
			paulis.append(pauli)
			coefficients.append(float(coefficient_string))

		assert all([len(p)==n for p in paulis])

		if params_dict['translation_invariant']:
			paulis, coefficients = addTranslates(n, paulis, coefficients, periodic_bc = params_dict['periodic'])

		params_dict['terms'] = paulis
		params_dict['coefficients'] = coefficients

	if len(lines[1].split(',')) == 2:
		assert lines[1].split(',')[0] == 'pauli'
		assert lines[1].split(',')[1] == 'coefficient\n'

		paulis = []
		coefficients = []
		for line in lines[2:]:
			line_split = line.split(',')
			pauli = line_split[0]
			coefficient_string = line_split[1]
			if coefficient_string[-1] == '\n':
				coefficient_string = coefficient_string[:-1]
			paulis.append(utils.decompressPauli(pauli,n))
			coefficients.append(float(coefficient_string))

		assert all([len(p)==n for p in paulis])

		if params_dict['translation_invariant']:
			paulis, coefficients = addTranslates(n, paulis, coefficients, periodic_bc = params_dict['periodic'])

		#sort the paulis to make it look nicer
		sorted_pairs = sorted(zip(paulis,coefficients), key = lambda x : utils.compressPauliToList(x[0]))
		paulis = [x[0] for x in sorted_pairs]
		coefficients = [x[1] for x in sorted_pairs]

		params_dict['terms'] = paulis
		params_dict['coefficients'] = coefficients

	return params_dict

#Hamiltonian is described by paulis_in and coefficients_in
#DEPRECATED
def computeGroundstate1D(n, paulis, coefficients, method = 'exact_diagonalization'):
	if method == 'exact_diagonalization':
		psi = exactDiagonalization(n,paulis,coefficients)
		out = lambda paulis : [computeExpectationFromWaveFunc(p,psi,n) for p in paulis]
	elif method == 'DMRG':
		out = computeGroundstateDMRG(n,paulis, coefficients)
	else:
		raise ValueError
	return out

def computeGroundstateDMRG(H, simulator_params):
	n = H.n
	ham_terms_tenpy = [pauliStringToTenpyTerm(n,pauli) for pauli in H.terms]
	model_params = dict(L=n, bc_MPS='finite', conserve=None, ham = (ham_terms_tenpy, H.coefficients))
	M = generalSpinHalfModel(model_params)
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

##TODO: NEW SIGNATURE (hamiltonian,beta, simulator_params)
def computeThermalStateByPurification(H, beta_max, simulator_params):
	hamiltonian_terms = H.terms
	coefficients = H.coefficients
	n = H.n

	dt = simulator_params['simulator_dt']
	if simulator_params['periodic'] == False:
		bc = 'finite'
	order = simulator_params['simulator_order']
	approx = simulator_params['simulator_approx']
	ham_terms_tenpy = [pauliStringToTenpyTerm(n,pauli) for pauli in hamiltonian_terms]
	model_params = dict(L=n, bc_MPS='finite', conserve = None, ham = (ham_terms_tenpy, coefficients))
	M = generalSpinHalfModel(model_params)
	psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
	options = {'trunc_params': {'chi_max': 100, 'svd_min': 1.e-8}}
	beta = 0.
	if order == 1:
		Us = [M.H_MPO.make_U(-dt, approx)]
	elif order == 2:
		Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
	eng = PurificationApplyMPO(psi, Us[0], options)
	betas = [0.]
	while beta < beta_max:
		beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
		betas.append(beta)
		for U in Us:
			eng.init_env(U)  # reset environment, initialize new copy of psi
			eng.run()  # apply U to psi
	return psi

#can be called from command line to generate a state and spit out a list of expectations.
#TODO: DELETE THIS?
if __name__ == '__main__':
	#python3 state_simulator.py 9 transverse_ising.txt out.txt -t -p  

	parser = argparse.ArgumentParser()
	parser.add_argument('n')
	parser.add_argument('hamiltonian_filename')
	parser.add_argument('-o', '--output_filename', help = 'default is hamiltonian_filename + _groundstate.txt')
	#parser.add_argument('-t', '--trans', action = 'store_true', help = 'is the input hamiltonian translation invariant')
	#parser.add_argument('-p', '--periodic', action = 'store_true', help = 'if translation-invariant, is it periodic')
	args_dict = vars(parser.parse_args())
	translation_invariant = args_dict['trans']
	periodic =  args_dict['periodic']
	n = int(args_dict['n'])
	if args_dict['output_filename'] is None:
		output_filename = args_dict['hamiltonian_filename'] + '_groundstate.txt'
	else:
		output_filename = args_dict['output_filename']

	H_paulis, H_coefficients, H_params = loadHamiltonian(args_dict['hamiltonian_filename'])
	psi = computeGroundstate1D(n, H_paulis, H_coefficients)
	saveState(n,psi, H_params['periodic'], output_filename)
