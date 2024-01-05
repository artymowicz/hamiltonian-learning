import numpy as np
import scipy
from utils import SparseTensor

def testSparseTensor(shape,indices,values, v):

	x = SparseTensor(shape, indices, values)
	print(f'x.shape = {x.shape}')
	print(f'x.indices = {x.indices}')
	print(f'x.values = {x.values}')
	print()

	print('removing zeros')
	x.removeZeros()
	print(f'x.shape = {x.shape}')
	print(f'x.indices = {x.indices}')
	print(f'x.values = {x.values}')
	print()

	if len(shape) == 2:
		y = x.toScipySparse()
		print('fx.toScipySparse() gives:')
		print(y.toarray())
		print()

	z = x.toNumpy()
	print('x.toNumpy() gives:')
	print(z)
	print()

	w = x.contractRight(v)
	print('x.contractRight(v) gives:')
	print(w.toNumpy())
	print('it should give:')
	print(z@v)
	print()

#valid input
shape = (2,2)
indices = [[0,1],[1,0],]
values = [2,3]
v = np.arange(2)

#testSparseTensor(shape,indices,values,v)

#valid input
shape = (5,5)
indices = [[1,2],[0,0], [4,3],[1,1]]
values = [0,2,3,2]
v = np.arange(5)

#testSparseTensor(shape,indices,values,v)

#valid input
shape = (5,5,3)
indices = [[1,2,1],[0,0,2], [4,3,2],[1,1,1]]
values = [0,2,3,2]
v = np.arange(3)

testSparseTensor(shape,indices,values,v)

#invalid input
shape = (2,2)
indices = [[0,1],[1,0],]
values = [2,3]
v = np.arange(3)

#invalid input
shape = (2,2)
indices = [[0,1],[1,0],]
values = [0,2,3,2]
v = np.arange(5)

#invalid input
shape = (5,5)
indices = [[1,2],[0,0], [5,3],[1,1]]
values = [0,2,3,2]
v = np.arange(5)







