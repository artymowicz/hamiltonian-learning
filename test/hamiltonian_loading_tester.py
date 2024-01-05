from state_simulator import Hamiltonian
import utils
import argparse

def saveLoadTest(ham_file):
	H = Hamiltonian('./hamiltonians/'+ham_file+'.txt')
	out_dir = './hamiltonians/disordered/'
	out_name = 'test.yml'

	H.saveToYAML(out_dir+out_name)

	H_2 = Hamiltonian(out_dir+out_name)

	assert H == H_2

def addDisorder(hamiltonian_name, out_name, disorder_magnitude):
	if out_name == None:
		out_name = hamiltonian_name + '_disordered'
	out_dir = './hamiltonians/'
	H = Hamiltonian('./hamiltonians/'+hamiltonian_name+'.txt')
	if disorder_magnitude >0:
		H.addDisorder(magnitude)
	H.saveToYAML(out_dir+out_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('hamiltonian_name')
	parser.add_argument('-o', '--out_name', default = None)
	parser.add_argument('-d','--disorder', type = float, default = 0)
	args = parser.parse_args()
	#saveLoadTest(ham_file)
	addDisorder(args.hamiltonian_name, args.out_name,args.disorder)