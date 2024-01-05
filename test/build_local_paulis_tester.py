import utils


n=20
k=2

paulis = utils.buildKLocalPaulis1D(n,k, False)

for i in range(len(paulis)):
	print(paulis[i])
	#print(utils.compressPauli(paulis[i]))