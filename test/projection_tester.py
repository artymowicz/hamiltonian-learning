import numpy as np
import matplotlib.pyplot as plt
import scipy
import argparse
import datetime
import os
import projection
import utils

def saveExpectations(save_dir, onebody_paulis, onebody_expectations, twobody_paulis, 
	twobody_expectations, twobody_expectations_theoretical, k):
	f = open(save_dir+ '/expectations.txt', "w")
	l_1 = len(onebody_paulis)
	l_2 = len(twobody_paulis)

	twobody_errors = twobody_expectations - twobody_expectations_theoretical
	f.write(f'two-body mse: {np.square(np.linalg.norm(twobody_errors))/l_2}\n\n')

	f.write('One body expectations\n\n')
	f.write('pauli' + ' '*(4*k-5) + '  :  expectation\n')
	f.write('-'*30+'\n')
	for i in range(l_1):
		pauli_compressed = utils.compressPauli(onebody_paulis[i])
		current_weight = int(pauli_compressed[0])
		pad = 4*(k - current_weight)
		f.write(pauli_compressed + ' '*pad + '  :  ' + f'{onebody_expectations[i]:+.5e}\n')

	f.write('\n\nTwo body expectations\n\n')
	f.write('pauli' + ' '*(8*k-5) + '  :  expectation proj  :  expectation theor : error\n')
	f.write('-'*30+'\n')
	for i in range(l_2):
		pauli_compressed = utils.compressPauli(twobody_paulis[i])
		current_weight = int(pauli_compressed[0])
		pad = 4*(2*k - current_weight)
		f.write(pauli_compressed + ' '*pad + '  :  ' + f'{twobody_expectations[i]:+.5e}' + ' '*5 + 
			'  :  ' + f'{twobody_expectations_theoretical[i]:.5e}' + ' '*5 +
			f'{twobody_errors[i]:.5e}' +'\n')
	f.close()

def plot(expectations_theoretical, expectations, save_dir):
	plt.scatter(expectations_theoretical, expectations, s = 2, marker='o')
	plt.plot(expectations_theoretical, expectations_theoretical, 'r')
	reg = scipy.stats.linregress(expectations_theoretical,expectations)
	x = expectations_theoretical
	m = reg.slope
	b = reg.intercept
	print(f'm = {m}, b = {b}')
	plt.plot(x, np.multiply(m,x)+b, c='g')
	plt.title(f'expectations values of correlators')
	plt.legend()
	plt.savefig(save_dir + f"/projected_expectations.pdf", dpi=150)
	plt.show()

identity = lambda x : x 
#keys are expected parameters in an experiment scheme and values are the function used to parse the corresponding value
param_conversions = {
					'target_script' : identity,
					'state_filename': identity,
					'state_type': identity, 
					'onebody_uncertainty_metric': identity,
					'objective_type': identity,
					'k': int,
					'onebody_uncertainty': float, 
					'geom_local': utils.strToBool,
					'transl_inv': utils.strToBool,
					'periodic_bc': utils.strToBool
					}

def loadExperimentScheme(filename):
	with open('./schemes/' + filename + '.txt') as f:
		lines = f.readlines()
	raw_params_dict = {}
	for line in lines:
		if line[0]=='#' or line[0]=='\n':
			continue
		line_split = line.split(':')
		if len(line_split)!=2:
			raise ValueError(f'Unable to parse line: {line}')
		raw_params_dict[line_split[0].strip()] = line_split[1][:-1].strip()
	
	params_dict = dict([(key, param_conversions[key](raw_params_dict[key])) for key in param_conversions.keys()])
	if params_dict['target_script'] != 'projection_tester.py':
		x = params_dict['target_script']
		raise ValueError(f'target script: {x} , expected projection_tester.py ')

	print('loading state')
	state_evaluator, n = utils.buildStateEvaluator(params_dict['state_filename'], params_dict['state_type'])

	if params_dict['geom_local']:
		onebody_paulis = utils.buildKLocalPaulis1D(n,params_dict['k'], params_dict['geom_local'])
	else:
		onebody_paulis = utils.buildPaulisUpToWeightW(n,params_dict['k'])

	return onebody_paulis, state_evaluator, params_dict

if __name__ == '__main__':
	#python3 tester.py sample_scheme

	parser = argparse.ArgumentParser()
	parser.add_argument('scheme_filename')
	parser.add_argument('-ns', '--nosave', action = 'store_true', help = 'skip creating a run directory')
	args = parser.parse_args()

	onebody_paulis, state_evaluator, params_dict = loadExperimentScheme(args.scheme_filename)

	print('computing one-body expectations')
	onebody_expectations = state_evaluator(onebody_paulis)

	twobody_paulis, twobody_expectations = projection.projection(onebody_paulis, onebody_expectations, params_dict)

	print('computing theoretical two-body expectations')
	twobody_expectations_theoretical = state_evaluator(twobody_paulis)

	if args.nosave:
		saveExpectations('.', onebody_paulis, onebody_expectations, twobody_paulis, 
		twobody_expectations, twobody_expectations_theoretical, params_dict['k'])
	else:
		save_dir = utils.createSaveDirectory()
		saveExpectations(save_dir, onebody_paulis, onebody_expectations, twobody_paulis, 
		twobody_expectations, twobody_expectations_theoretical, params_dict['k'])
		os.system(f'cp ./schemes/{args.scheme_filename}.txt {save_dir}/{args.scheme_filename}.txt')

	plot(twobody_expectations_theoretical, twobody_expectations, save_dir)



