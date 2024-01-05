import numpy as np
import scipy
import cvxpy as cp
import argparse
import datetime
from utils import saveExpectations, loadExpectations, invertStringList, multiplyPaulis
'''
#TODO: change everything so that the identity operator is not treated differently
def buildMultiplicationTensor(onebody_operators):
	print('building multiplication tensor')
	n = len(onebody_operators[0])
	assert ''.join('I'*n) not in onebody_operators

	onebody_operators_augmented = [''.join('I'*n)] + onebody_operators
	R = len(onebody_operators_augmented)
	row_indices = []
	column_indices = []
	values = []

	twobody_operators = []
	twobody_indices_dict = {}
	l = 0

	for i in range(R):
			for j in range(i+1,R):

				W,z = multiplyPaulis(onebody_operators_augmented[i],onebody_operators_augmented[j])

				if W not in twobody_indices_dict:
					twobody_operators.append(W)
					l += 1
					twobody_indices_dict[W] = l-1

				row_indices.append(R*i+j)
				column_indices.append(twobody_indices_dict[W])
				values.append(z)

				row_indices.append(R*j+i)
				column_indices.append(twobody_indices_dict[W])
				values.append(np.conjugate(z))

	mult_tensor = scipy.sparse.coo_array((values,(row_indices,column_indices)), shape = (R**2,l))

	return mult_tensor, twobody_operators
'''

def createSaveDirectory():
	now = datetime.datetime.now()
	dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
	dirname = f'./runs/{dt_string}'
	os.mkdir(dirname)
	return dirname

#given two lists l_1,l_2 such that l_1 is contained in l_2, returns indices of l_1 and indices of its complement
def embedded_list_indices(l_1,l_2):
	indices_dict = invertStringList(l_2)
	l_1_indices = [indices_dict[s] for s in l_1]

	l_1_complement_indices = []
	for i in range(len(l_2)):
		if i not in l_1_indices:
			l_1_complement_indices.append(i)
	return l_1_indices, l_1_complement_indices

#vectorizes the first two axes of a rank-3 tensor using Fortran ordering
def vectorize(t):
	assert len(t.shape) == 3
	(a,b,c) = t.shape
	print(f'(in vectorize function): t.shape = {t.shape}')
	t = np.reshape(t, (a*b, c), order = 'F')
	return t

def coo_inclusion_matrix(indices, ambient_dim):
	l = ambient_dim
	data = np.ones(len(indices))
	col = range(len(indices))
	return scipy.sparse.coo_array((data, (indices, col)), shape=(l, len(indices)))

'''
Mandatory parameters in params_dict:
transl_inv
onebody_uncertainty
onebody_uncertainty_metric
objective_type

Optional parameters in params_dict:
objective_vector
'''
def projection(onebody_operators, onebody_expectations, params_dict):
	assert onebody_operators[0] == 'I'*len(onebody_operators[0])

	print('computing projection of two-body operators')
	transl_inv = params_dict['transl_inv']
	onebody_uncertainty = params_dict['onebody_uncertainty']
	onebody_uncertainty_metric = params_dict['onebody_uncertainty_metric']
	objective_type = params_dict['objective_type']
	
	if transl_inv == False:
		mult_tensor, twobody_operators = utils.buildMultiplicationTensor(onebody_operators)
		mult_tensor.vectorize([0,1])
		mult_tensor = mult_tensor.toScipySparse()
	else:
		print('havent implemented translation_invariant mult_tensor') #TODO
		return

	r = len(onebody_operators)
	l = len(twobody_operators)
	
	fixed_variable_indices, decision_variable_indices = embedded_list_indices(onebody_operators, twobody_operators)

	if onebody_uncertainty != 0:
		print('havent implemented case where onebody_uncertainty isnt zero')
		return

	fixed_variables_inclusion = coo_inclusion_matrix(fixed_variable_indices, l)
	decision_variables_inclusion = coo_inclusion_matrix(decision_variable_indices, l)
	
	mult_tensor_fixed_variables = mult_tensor@fixed_variables_inclusion
	mult_tensor_decision_variables = mult_tensor@decision_variables_inclusion

	X = cp.Variable(l-r)

	if 'objective_vector' in params_dict:
		raise ValueError('havent implemented what to do when objective_parameters_dict isnt None') #TODO
	else:
		X_objective = np.random.normal(size = (l-r,))

	F = lambda x,y : np.eye(r+1) + cp.reshape(mult_tensor_fixed_variables@x + mult_tensor_decision_variables@y, (r+1,r+1))
	constraints = [F(onebody_expectations, X) >> 0]
	if objective_type == 'linear':
		objective = cp.sum(cp.multiply(X,X_objective))
	elif objective_type == 'quadratic':
		objective = cp.sum(cp.square(X-X_objective))
	else:
		raise ValueError(f'unknown objective type {objective_type}')
	prob = cp.Problem(cp.Minimize(objective), constraints)
	prob.solve(solver = 'MOSEK', verbose = True)#, mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':10})
	#prob.solve(solver = 'SCS', verbose = True, eps_abs = 1e-4, eps_rel = 1e-4)
	solution = fixed_variables_inclusion@onebody_expectations + decision_variables_inclusion@X.value

	return twobody_operators, solution

if __name__ == '__main__':
	#python3 projection.py 9_veryrandom_nonperiodic_expectations.txt out.txt

	parser = argparse.ArgumentParser()
	parser.add_argument('onebody_expectations_filename')
	parser.add_argument('output_filename')
	args_dict = vars(parser.parse_args())

	onebody_operators, onebody_expectations = loadExpectations(args_dict['onebody_expectations_filename'])
	onebody_expectations_dict = dict(zip(onebody_operators, onebody_expectations))
	twobody_operators, twobody_expectations = projection(onebody_expectations_dict)

	saveExpectations(twobody_operators, twobody_expectations, args_dict['output_filename'])
