import utils
import itertools

n = 2

single_site = ['I','X','Y','Z']
single_site_repeated = tuple([single_site]*n)

paulis = [''.join(l) for l in itertools.product(*single_site_repeated)]

pretty_cx_nums = {1+0j: '',-1+0j: '-',  -1j: '-i', 1j: 'i'}

for p_1,p_2 in itertools.product(paulis, paulis):
	W,w = utils.multiplyPaulis(p_1,p_2)
	print(f'{p_1} {p_2} = {pretty_cx_nums[w]}{W}')