from simulation import Hamiltonian, Simulator
import hamiltonian_learning
import utils

import os
import pickle
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import yaml
import argparse
from matplotlib import colormaps

def generateOperators(params):
	n = params['n']
	k = params['k']
	periodic_bc = params['periodic']
	utils.tprint('computing onebody operators')
	if params['onebody_locality'] == 'short_range':
		onebody_operators = utils.buildKLocalPaulis1D(n, k, periodic_bc)
	elif params['onebody_locality'] == 'long_range':
		onebody_operators = utils.buildKLocalCorrelators1D(n,params['k'], periodic_bc, d_max = params['onebody_range'])
	else:
		raise ValueError(f"unrecognized value for onebody_locality: {params['onebody_locality']}")
	
	utils.tprint('computing Hamiltonian terms')
	if params['H_locality'] == 'short_range':
		hamiltonian_terms = utils.buildKLocalPaulis1D(n, params['k_H'], periodic_bc)
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

def main(params):
	utils.tprint(f'running modular_tester.py with params:')
	print(yaml.dump(params))

	### load Hamiltonian and give it a descriptive name
	H, H_name = loadAndNameHamiltonian(params)

	### generate operators used in convex optimization
	_, _, threebody_operators = generateOperators(params)

	### initialize state
	if type(params['beta']) == float or type(params['beta']) == int:
		beta = params['beta']
	elif params['beta'] == 'inf':
		beta = np.inf
	else:
		raise ValueError
	state = Simulator(params['n'],H,H_name,beta)

	#threebody_expectations = np.flip(state.getExpectations(np.flip(threebody_operators), params))
	threebody_expectations = np.flip(state.getExpectations(np.flip(threebody_operators), params))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('setup_filename')
	parser.add_argument('-g', type = float)
	parser.add_argument('-b', '--beta', type = float)
	parser.add_argument('-dt', '--simulator_dt', type = float)
	parser.add_argument('-n', type = int)
	args = parser.parse_args()

	##load params dictionary
	with open(args.setup_filename) as f:
		params = yaml.safe_load(f)

	## some parameters can be provided in the commandline, they will then override the ones in the setup file
	if args.g is not None:
		params['coupling_constants']['g'] = args.g
	if args.beta is not None:
		params['beta'] = args.beta
	if args.simulator_dt is not None:
		params['simulator_dt'] = args.simulator_dt
	if args.n is not None:
		params['n'] = args.n

	main(params)

