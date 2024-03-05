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
import matplotlib.pyplot as plt

from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO

### Parameter used to determine n_chunks in DFScomputeParallel. 
### Generally want it larger than 10, and CHUNKS_PER_THREAD * expectations_n_threads << total number of threebody operators
CHUNKS_PER_THREAD = 16

### In getExpectationValues, if some cached expectation values are missing, all expectation values are recomputed
### In this case, we check that cached expectations agree with computed expectations
### This parameter determines the tolerance used in this equality check 
CACHE_TOLERANCE = 1e-10

### When computing the thermal state at a given inverse temperature beta, 
### some of the intermediate states at inverse temperature (0,beta) are cached
### This parameter determines which ones are cached
BETA_SAVE_INTERVAL = 1e-1

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

	def plot(self, title, save_filename, skip_plotting, normalization = 1):
		thresh = normalization*1e-3
		fig = plt.figure(1)
		ax = fig.add_subplot(111)

		coeffs_dict = dict(zip(self.terms, self.coefficients))
		l = ['X','Y','Z']
		single_site_coeffs = {}
		for A in l:
			single_site_coeffs[A] = np.zeros(self.n)
			for k in range(self.n):
				p = utils.decompressPauli(f'1 {A} {k}',self.n)
				if p in coeffs_dict:
					single_site_coeffs[A][k] = normalization*coeffs_dict[p]

			if np.max(np.abs(single_site_coeffs[A])) > thresh:
				ax.scatter(np.arange(self.n), single_site_coeffs[A], s=2, label = f'{A}')

		AB_learned_coeffs = {}
		for (A,B) in itertools.product(l,repeat = 2):
			AB_learned_coeffs[(A,B)] = np.zeros(self.n-1)
			for k in range(self.n-1):
				p = utils.decompressPauli(f'2 {A} {k} {B} {k+1}',self.n)
				if p in coeffs_dict:
					AB_learned_coeffs[(A,B)][k] = normalization*coeffs_dict[p]
			if np.max(np.abs(AB_learned_coeffs[(A,B)])) > thresh:
				ax.scatter(np.arange(self.n-1)+0.5, AB_learned_coeffs[(A,B)], s=2, label = f'{A}{B}')

		AIB_learned_coeffs = {}
		for (A,B) in itertools.product(l,repeat = 2):
			AIB_learned_coeffs[(A,B)] = np.zeros(self.n-2)
			for k in range(self.n-2):
				p = utils.decompressPauli(f'2 {A} {k} {B} {k+2}',self.n)
				if p in coeffs_dict:
					AIB_learned_coeffs[(A,B)][k] = normalization*coeffs_dict[p]
			if np.max(np.abs(AIB_learned_coeffs[(A,B)])) > thresh:
				ax.scatter(np.arange(self.n-2)+1, AIB_learned_coeffs[(A,B)], s=2, label = f'{A}I{B}')

		ax.set_xlabel('site')
		ax.set_title(title)
		#ax.set_xlim(right=self.n+10)
		fig.legend(loc='outside right center')
		fig.savefig(save_filename, dpi=150)
		if skip_plotting is False:
			plt.show()

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
		if 'I'*self.n not in self.terms:
			self.terms = np.concatenate((['I'*self.n],self.terms))
			self.coefficients = np.concatenate(([1.],self.coefficients))
		self.sort()
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

	### MAY REFUSE TO WORK IF periodic = True AND self.terms CONTAINS TERMS OF RANGE MORE THAN n/2 TODO:FIX or make it throw an exception
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

	def loadHamiltonian(self,n,filename, couplings = {}):
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

### tenpy model that is used to compute thermal & ground states
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


class Simulator:
	def __init__(self, n, H, H_name, beta):
		self.n = n
		self.H = H
		self.H_name = H_name
		self.beta = beta
		self.psi = None
		self.metrics = {}

	def computeEquilibriumStateMPS(self, params, return_intermediate_results = False):
		if params['simulator_method'] == 'tenpy':
			t1 = time.time()
			if self.beta == np.inf:
				utils.tprint('computing state using DMRG')
				psi = computeGroundstateDMRG(self.H, params)
				t2 = time.time()
				self.metrics['rho_computation_time_by_DMRG'] = t2-t1
			else:
				utils.tprint('computing state using purification')
				betas, psis = computeThermalStateByPurification(self.H, self.beta, params)
				psi = psis[-1]
				t2 = time.time()
				self.metrics['rho_computation_time_by_MPO'] = t2-t1
		else:
			raise ValueError(f"invalid value of parameter simulator_method: {params_['simulator_method']} (expected 'tenpy'))")

		if return_intermediate_results:
			if params['simulator_method'] == 'tenpy' and self.beta < np.inf:
				return betas, psis
			else:
				raise ValueError('no intermediate psis generated')
		else:
			return psi

	def saveTenpyMPS(self,path, psi, params):
		psi_params = {}
		for key in params:
			if key[:10] == 'simulator_':
				psi_params[key] = params[key]

		data = {"psi": psi,  
		"parameters": psi_params}

		if self.beta < np.inf and params['MPS_avoid_overwrite'] and os.path.exists(path):
			try:
				with h5py.File(path, 'r') as f:
					try:
						existing_cache_params = f['parameters']
						cached_dt = existing_cache_params['simulator_dt'][()]
						if params['simulator_dt'] > cached_dt:
							utils.tprint(f"avoided saving over {path} because it was computed with dt = {cached_dt}")
							return
					except KeyError:
						utils.tprint(f"warning: KeyError when checking existing cache at {path}")
			except OSError:
				utils.tprint(f"warning: OSError when checking existing cache at {path}. overwriting")

		with h5py.File(path, 'w') as f:
			tenpy.tools.hdf5_io.save_to_hdf5(f, data)

	def loadTenpyMPS(self,path):
		with h5py.File(path, 'r') as f:
			data = tenpy.tools.hdf5_io.load_from_hdf5(f)
		psi = data['psi']
		psi_params = data['parameters']
		return psi, psi_params

	def getEquilibriumStateMPS(self, params):
		if params['MPS_no_cache']:
			return self.computeEquilibriumStateMPS(params, return_intermediate_results = False)

		cache_filename = f"{self.H_name}__b={self.beta:.10f}__state.hdf5"
		cache_directory = './caches/mps/'
		cache_path = cache_directory + cache_filename

		### load the state from a cache if possible

		if params['MPS_overwrite_cache'] == False:
			loading = False
			if not os.path.exists(cache_path):
				if params['printing_level'] > 2:
					utils.tprint(f'mps cache {cache_filename} not found')
			else:
				if params['printing_level'] > 2:
					utils.tprint(f'mps cache {cache_filename} found')
				
				passed_sanity_check = False
				with h5py.File(cache_path, 'r') as f:
					if False: ### TODO: put some sanity checks here if needed
						pass
					else:
						passed_sanity_check = True
				if passed_sanity_check:
					loading = True
				else:
					tmp_file_path = cache_directory + 'tmp/' + cache_filename
					utils.tprint(f'unable to read mps cache at {cache_path}. moving old cache to {tmp_file_path} and creating new one')
					
					if not os.path.exists(cache_directory + 'tmp/'):
							os.mkdir(cache_directory + 'tmp/')
					os.replace(cache_path, tmp_file_path)

			if loading:
				psi, psi_params = self.loadTenpyMPS(cache_path)
				utils.tprint(f"mps state loaded from {cache_path}. params:")
				print()
				print(yaml.dump(psi_params))
				return psi

		### if execution reaches this line, the state was not loaded and will be computed

		if self.beta < np.inf:
			betas, psis = self.computeEquilibriumStateMPS(params, return_intermediate_results = True)
			if round(betas[-1],10) != round(self.beta,10):
				utils.tprint(f'warning: highest beta was not computed betas[-1] = {betas[-1]}, self.beta = {self.beta}')

			if params['printing_level'] > 1:
				utils.tprint('saving states')
			if not os.path.exists(cache_directory):
					os.mkdir(cache_directory)

			for i in range(len(betas)):
				save_path = cache_directory + f"{self.H_name}__b={betas[i]:.10f}__state.hdf5"
				if True:#round(betas[i]%BETA_SAVE_INTERVAL,10) in (0, BETA_SAVE_INTERVAL):
					if params['printing_level'] > 2:
						utils.tprint(f'saving ' + save_path)
					self.saveTenpyMPS(save_path, psis[i], params)
			return psis[-1]
		else:
			psi = self.computeEquilibriumStateMPS(params, return_intermediate_results = False)
			if params['printing_level'] > 1:
				utils.tprint('saving state')
			if not os.path.exists(cache_directory):
					os.mkdir(cache_directory)

			save_path = cache_directory + f"{self.H_name}__b={self.beta:.10f}__state.hdf5"
			utils.tprint(f'saving ' + save_path)
			self.saveTenpyMPS(save_path, psi, params)
			return psi

	### Assumes that operators are sorted in alphabetical order
	def computeExpectations(self, operators, params):
		if self.psi == None and params['simulator_method'] != 'ED':
			self.psi = self.getEquilibriumStateMPS(params)

		if params['simulator_method'] == 'tenpy':
			n_chunks = params['expectations_n_threads']*CHUNKS_PER_THREAD
			t1 = time.time()

			if self.beta == np.inf:
				state_type = 'pure'
			else:
				state_type = 'mixed'

			### we flip the list of operators when passing to DFSComputeParallel because it requires the list to be in reverse alphabetical order
			args = (self.n, self.psi, np.flip(operators), state_type, params['expectations_n_threads'], n_chunks)
			kwargs = dict(naive = params['expectations_naive_compute'])
			expectations, tenpy_calls = DFScomputeParallel(*args, **kwargs)
			expectations = np.flip(expectations)

			t2 = time.time()
			self.metrics['tenpy_calls'] = tenpy_calls
			self.metrics['expectations_computation_time'] = t2-t1

		elif params['simulator_method'] == 'ED':
			expectations = self.computeExpectationsED(operators)
		else:
			raise ValueError(f"unrecognized value of parameter simulator_method: {params['simulator_method']}")

		return expectations

	def getExpectations(self, operators, params):
		if np.abs(params['disorder']) > 0 and not (params['expectations_no_cache'] and params['MPS_no_cache']):
			if params['printing_level'] > 2:
				utils.tprint("Skipping all caching because Hamiltonian has nonzero disorder")
			params['expectations_no_cache'] = True
			params['MPS_no_cache'] = True

		if params['expectations_no_cache']:
			return self.computeExpectations(operators, params)

		if not os.path.exists('./caches/'):
			os.mkdir('./caches/')

		filename = f"{self.H_name}__b={self.beta:.10f}.hdf5"

		new_cache = True
		if not os.path.exists('./caches/'+filename):
			if params['printing_level'] > 2:
				utils.tprint(f'expectations cache {filename} not found')
		else:
			if params['printing_level'] > 2:
				utils.tprint(f'expectations cache {filename} found')
			### some sanity checks on the cache
			passed_sanity_check = False
			with h5py.File(f'./caches/{filename}', 'r') as exp_file:
				if '/hamiltonian/terms' not in exp_file:
					message = f'warning: dataset /hamiltonian/terms not found in cache at ./caches/{filename}'
				elif '/hamiltonian/coeffs' not in exp_file:
					message = f'warning: dataset /hamiltonian/coeffs not found in cache at ./caches/{filename}'
				elif '/expectations/operators' not in exp_file:
					message = f'warning: dataset /expectations/operators not found in cache at ./caches/{filename}'
				elif '/expectations/exp_values' not in exp_file:
					message = f'warning: dataset /expectations/exp_values not found in cache at ./caches/{filename}'
				elif not np.array_equal(np.char.decode(exp_file['/hamiltonian/terms']), self.H.terms):
					message = f'warning: cached expectations hamiltonian terms do not agree with current hamiltonian'
				elif not np.array_equal(exp_file['/hamiltonian/coeffs'], self.H.coefficients):
					message = f'warning: cached expectations hamiltonian coefficients do not agree with current hamiltonian'
				else:
					passed_sanity_check = True
					message = f'cache {filename} passed sanity checks'

			if (passed_sanity_check == False and params['printing_level'] > 0) or (passed_sanity_check == True and params['printing_level'] > 2):
				utils.tprint(message)

			if passed_sanity_check:
				new_cache = False
			else:
				tmp_file_path = f'./caches/tmp/{filename}'
				if params['printing_level'] > 0:
					utils.tprint(f'unable to read expecations cache at ./caches/{filename}. moving old cache to {tmp_file_path}')
				if not os.path.exists('./caches/tmp/'):
						os.mkdir('./caches/tmp/')
				os.replace(f'./caches/{filename}', tmp_file_path)
			
		if new_cache:
			if params['printing_level'] > 2:
				utils.tprint(f'creating expectations cache {filename}')
			with h5py.File(f'./caches/{filename}', 'w') as exp_file:
				exp_file['/hamiltonian/terms'] = np.char.encode(self.H.terms)
				exp_file['/hamiltonian/coeffs'] = self.H.coefficients
				exp_file.create_group('/expectations/')

		else:
			with h5py.File(f'./caches/{filename}', 'r') as exp_file:
				if params['printing_level'] > 2:
					utils.tprint(f'decoding strings')
				cached_operators = np.char.decode(exp_file[f'expectations/operators'])
				cached_expectations = exp_file[f'expectations/exp_values']
				cached_expectations_dict = dict(zip(cached_operators, cached_expectations))

			if params['expectations_skip_checks'] and not params['expectations_overwrite_cache']:
				return np.array([cached_expectations_dict[p] for p in operators])

			if params['printing_level'] > 2:
				utils.tprint(f'checking that all required expectations are cached')
			all_there = True
			operators_set = set(cached_operators)

			for t in operators:
				if t not in operators_set:
					if params['printing_level'] > 1:
						utils.tprint(f'operator {t} not found in cache')
					all_there = False
					break

			if all_there:
				if params['printing_level'] > 2:
					utils.tprint(f'all required expectations found in cache')
				if not params['expectations_overwrite_cache']:
					if params['printing_level'] > 0:
						utils.tprint(f'expectations loaded from cache ./caches/{filename}')
					return np.array([cached_expectations_dict[p] for p in operators])

		if params['printing_level'] > 2:
			utils.tprint('computing expectation values using method '+ params['simulator_method'])
		computed_expectations = self.computeExpectations(operators, params)
		computed_expectations_dict = dict(zip(operators, computed_expectations))

		if not new_cache:
			if params['printing_level'] > 2:
				utils.tprint('checking consistency of newly computed expectation values with old ones')
			max_difference = np.max(np.abs([cached_expectations_dict[o]-computed_expectations_dict[o] for o in operators if o in cached_expectations_dict]))
			if max_difference < CACHE_TOLERANCE:
				cached_expectations_dict.update(computed_expectations_dict)
				total_operators = np.array(list(cached_expectations_dict.keys()))
				total_operators.sort()
				total_expectations = np.array([cached_expectations_dict[p] for p in total_operators])
			else:
				if not os.path.exists('./caches/tmp/'):
					os.mkdir('./caches/tmp/')
				tmp_file_path = f'./caches/tmp/{filename}'
				if params['printing_level'] > 0:
					utils.tprint(f'some expectations were not consistent with previously computed expectations (max difference = {max_difference}).'
						+ f' moving old cache to {tmp_file_path} and overwriting')
				if not os.path.exists('./caches/tmp/'):
						os.mkdir('./caches/tmp/')
				os.system(f'cp ./caches/{filename} tmp_file_path')

				sort_indices = np.argsort(operators)
				total_operators = operators[sort_indices]
				total_expectations = computed_expectations[sort_indices]
		else:
			sort_indices = np.argsort(operators)
			total_operators = operators[sort_indices]
			total_expectations = computed_expectations[sort_indices]
		if params['printing_level'] > 2:
			utils.tprint(f'saving new expectation values in ./caches/{filename}')
		with h5py.File(f'./caches/' + filename, 'r+') as cache:

			if 'operators' in cache[f'expectations'].keys():
				del cache[f'expectations/operators']
			cache[f'expectations/operators'] = np.char.encode(total_operators)

			if 'exp_values' in cache[f'expectations'].keys():
				del cache[f'expectations/exp_values']
			cache[f'expectations/exp_values'] = total_expectations

		return computed_expectations

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
		return np.array(out)

### assumes that operators are in REVERSE alphabetical order
def DFScomputeParallel(n, psi, operators, state_type, n_threads, n_chunks, naive = False):
	if n_threads == 1:
		if naive:
			return DFScomputeSingleThreadNaive(n,psi,operators)
		else:
			if state_type == 'pure':
				return DFScomputeSingleThreadPure(n,psi,operators)
			elif state_type == 'mixed':
				return DFScomputeSingleThreadMixed(n,psi,operators)
			else:
				raise ValueError

	l = len(operators)
	out = np.zeros(l)
	ind_list = [(l//n_chunks)*k for k in range(n_chunks)]
	ind_list += [l]
	operators_chunks = [operators[ind_list[i]:ind_list[i+1]] for i in range(n_chunks)]

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

### assumes that operators are in REVERSE alphabetical order
def DFScomputeSingleThreadPure(n, psi, operators):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*(n+1)
	L_tensors[0] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vR', 'vR*'])

	tenpy_calls = 0

	### compute R_tensors
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
		tenpy_calls += 1

	for i in range(l):
		p = operators[i]

		### get the first index that differs from the previous p
		for j in range(n):
			if p[j] != previous[j]:
				break

		### get the last nontrivial index of p (defaults to -1 if p is the identity)
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

	return out, tenpy_calls

def DFScomputeSingleThreadNaive(n,psi,operators):
	return np.array([psi.expectation_value_term(pauliStringToTenpyTerm(n,p)) for p in operators]), len(operators)

### assumes that operators are in REVERSE alphabetical order
def DFScomputeSingleThreadMixed(n, psi, operators):
	l = len(operators)
	out = np.zeros(l)
	previous = '-'*n
	L_tensors = [None]*(n+1)
	L_tensors[0] = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(np.asarray([[1]]), dtype=complex, labels=['vR', 'vR*'])

	tenpy_calls = 0

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
		tenpy_calls += 1

	for i in range(l):
		p = operators[i]

		###  get the first index that differs from the previous p
		for j in range(n):
			if p[j] != previous[j]:
				break

		### get the last nontrivial index of p (defaults to -1 if p is the identity)
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

				### METHOD 2
				tensors = dict(zip(tensor_names,tensor_list))
				tmp1 = tenpy.linalg.np_conserved.tensordot(L, tensors['B*'], axes = [['vR*'],['vL*']])
				#print(f'tmp1.shape = {tmp1.shape}')
				tmp2 = tenpy.linalg.np_conserved.tensordot(tensors['B'], tensors['O'], axes = [['p'],['p']])
				#print(f'tmp2.shape = {tmp2.shape}')
				L_tensors[k+1] = tenpy.linalg.np_conserved.tensordot(tmp1, tmp2, axes = [['vR','q*','p*'],['vL','q','p*']])
				tenpy_calls += 3

		out[i] = np.real(tenpy.algorithms.network_contractor.contract([L_tensors[y+1], R_tensors[y+1]], ['L', 'R'], [['L','vR','R','vL'], ['L','vR*','R','vL*']]))
		tenpy_calls += 1
		previous = p

	return out, tenpy_calls

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

def generatePauliMatrix(pauli_string):
	pauli_list = [scipy.sparse.coo_array(pauli_generators[c]) for c in pauli_string]
	return ft.reduce(scipy.sparse.kron, pauli_list)

### given a (left-justified) pauli returns all its translates
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

### converts a string of the form "key_1 : val_1, key_2 : val_2" to a dict
def stringToDict(s):
	d = {}
	s_split = s.split(', ')
	for x in s_split:
		x_split = x.split(' : ')
		assert len(x_split) == 2
		d[x_split[0]] = x_split[1]
	return d

identity = lambda x : x 
def strToBool(s):
	if s == 'True' or s == 'T' or s == 't' or s == '1':
		return True
	elif s == 'False' or s == 'F' or s == 'f' or s == '0':
		return False
	else:
		raise ValueError
### keys are parameters loaded in loadHamiltonian function and values are the function used to parse the corresponding value
param_conversions = {
					'n' : int,
					'k' : int,				
					'translation_invariant' : strToBool,
					'periodic' : strToBool,
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

def computeThermalStateByPurification(H, beta_max, simulator_params, final_state_only=False):
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
	if final_state_only == False:
		betas = []
		psis = []

	for beta in tqdm(np.arange(0,beta_max+2*dt, 2*dt)):# factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
		if final_state_only == False and round(beta%BETA_SAVE_INTERVAL,10) in (0,BETA_SAVE_INTERVAL):
			betas.append(beta)
			psis.append(psi.copy())
		for U in Us:
			eng.init_env(U)  # reset environment, initialize new copy of psi
			eng.run()  # apply U to psi
	if final_state_only == False:
		betas.append(beta)
		psis.append(psi.copy())
	if final_state_only:
		return psi
	else:
		return betas, psis
