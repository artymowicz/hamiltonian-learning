import utils

n = 8
k = 1
periodic_bc = False

for p in utils.buildKLocalCorrelators1D(n,k, periodic_bc, d_max = 1):
	print(p)