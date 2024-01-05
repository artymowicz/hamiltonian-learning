import numpy as np
import timeit
import math
import itertools
import random
import argparse

from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg

import hamiltonian_learning
import hamiltonian_learning_tester 
import utils
import state_simulator

import os
import sys

def DMRG_tf_ising_finite(L,J, g):
    #print(f'solving g={g} J = {J} TFI model with DMRG')
    model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'combine': True
    }
    dmrg.run(psi, M, dmrg_params)
    return psi

tenpy_paulis = ['Id','Sigmax','Sigmay','Sigmaz']
my_paulis = ['I','X','Y','Z']
tenpy_to_my = dict(zip(tenpy_paulis,my_paulis))
my_to_tenpy = dict(zip(my_paulis,tenpy_paulis))

'''
def pauliStringToTerm(n,p):
    assert len(p) == n 
    if p == 'I'*n:
        return [('Id',0)]
    term = []
    for i in range(n):
        if p[i] != 'I':
            term.append((my_to_tenpy[p[i]], i))
    return term
'''
'''
#these are of the form <AB> where A is a 2-local operator and B is a 3-local operator
def state_simulator.computeThreeBodyTermsAbridged(n, psi):
    counter = 0
    for x,y in itertools.product(twosite_terms, threesite_terms):
        for i in range(1):
            nonoverlapping_j = [j for j in range(n-2) if j>i+1]
            if len(nonoverlapping_j)>0:
                corr = psi.term_correlation_function_right(x, y, i_L=i, j_R=nonoverlapping_j)
'''

def timeN(n_min,n_max,interval,g):
    for n in range(n_min,n_max,interval):
        _,psi,_ = DMRG_tf_ising_finite(L=n, g=g)
        print(f'computing correlations for n = {n}')
        t.append((n,g, timeit.timeit(lambda: state_simulator.computeThreeBodyTermsAbridged(n,psi), number=1)))
        print(f'g : {g}, time: {t[-1][1]}')
    for x in t:
        print(f'n : {x[0]}, g : {x[1]}, execution time : {x[2]}')

def timeG(n, g_1, g_2, interval):
    times = []
    for g in np.arange(g_1,g_2, interval):
        psi = DMRG_tf_ising_finite(L=n, g=g)
        print(f'computing correlations for n = {n}')
        times.append((n,g, timeit.timeit(lambda: state_simulator.computeThreeBodyTermsAbridged(n,psi), number=1)))
        print(f'g : {g}, time: {times[-1][2]}')
    for x in times:
        print(f'n : {x[0]}, g : {x[1]}, execution time : {x[2]}')


class generalSpinHalfModel(CouplingMPOModel):
    default_lattice = Chain
    force_default_lattice = True
    r"""Transverse field Ising model on a general lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^x_i \sigma^x_{j}
            - \sum_{i} \mathtt{g} \sigma^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: TFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.

    """
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
        '''
        J = 1.
        g = 1.
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
        '''
        for term,coefficient in zip(hamiltonian_terms, coefficients):
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

def computeGroundstateDMRG(n,hamiltonian_terms, coefficients):
    '''
    lattice = Chain(n, SpinHalfSite())
    M = CouplingModel(lattice)
    for pauli,coefficient in zip(hamiltonian_terms, coefficients):
        term = pauliStringToTerm(n,pauli)
        if len(term) == 1:
            j = term[0][1]
            op = term[0][0]
            M.add_onsite_term(coefficient, j, op)
        elif len(term) == 2:
            j,k = term[0][1], term[1][1]
            op_j, op_k = term[0][0], term[1][0]
            M.add_coupling_term(coefficient, j, k, op_j, op_k)
        else:
            ijkl = [x[1] for x in term]
            ops_ijkl = [x[0] for x in term]
            M.add_multi_coupling_term(coefficient, ijkl, ops_ijkl, 'Id')
    M_MPO_model = MPOModel(lattice, M.calc_H_MPO())
    '''
    ham_terms_tenpy = [pauliStringToTerm(n,pauli) for pauli in hamiltonian_terms]
    model_params = dict(L=n, bc_MPS='finite', conserve=None, ham = (ham_terms_tenpy, coefficients))
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

### For 2-local Hamiltonians
def testAgainstED(paulis, coefficients, params):
    n = params['n']
    k = params['k']

    ## for now it works only for such parameters
    assert n>4
    assert k == 2

    '''
    paulis = ['XX'+ 'I'*(n-2), 'Z'+ 'I'*(n-1)]
    coefficients = [-J, -g]
    paulis, coefficients = state_simulator.addTranslates(n, paulis, coefficients, False)
    '''

    '''
    ### MPS
    print('computing groundstate by DMRG')
    psi = state_simulator.computeGroundstate1D(n, paulis, coefficients, method = 'DMRG')
    print('computing groundstate expectations from MPS')
    MPS_expectations_dict = state_simulator.computeThreeBodyTerms(n, psi)

    ### ED
    print('computing groundstate by ED')
    state_evaluator = state_simulator.computeGroundstate1D(n, paulis, coefficients, method = 'exact_diagonalization')
    onebody_operators = utils.buildKLocalPaulis1D(n,k, False)
    ED_expectations_dict = hamiltonian_learning_tester.computeAllRequiredExpectations(onebody_operators, onebody_operators, state_evaluator)
    '''

    simulator_params = dict(n = n,
                            k = k, 
                            hamiltonian_terms = paulis, 
                            hamiltonian_coefficients = coefficients,
                            hamiltonian_type = params['locality'])

    simulator = state_simulator.GroundstateSimulator1D(simulator_params)

    print('computing groundstate expectations by exact diagonalization')
    ED_expectations_dict = simulator.computeExpectations(method = 'exact_diagonalization')
    print('computing groundstate expectations by dmrg')
    MPS_expectations_dict = computeRequiredExpectations(h, onebody_operators, beta, simulator_params, method='DMRG', skip_intermediate_betas=True)

    ED_operators = sorted(list(ED_expectations_dict.keys()), key = utils.compressPauliToList)
    MPS_operators = sorted(list(MPS_expectations_dict.keys()), key = utils.compressPauliToList)

    if ED_operators != MPS_operators:
        print('ED operators dont match mps operators')

    ### compare m random expectation values
    m=20
    print(f'Comparison of {m} random expectation values:')
    first_line = ' pauli ' + ' '*(n-6) + ' : exp (ED)' + ' : expectation (DMRG)'
    print(first_line)
    print('-'*len(first_line))
    for p in [ED_operators[random.randint(0, len(ED_operators)-1)] for i in range(m)]:
        print(f' {p} : {ED_expectations_dict[p]:+.4f} '
            + f' : {MPS_expectations_dict[p]:+.4f}')

    expectation_difference = [MPS_expectations_dict[key]-ED_expectations_dict[key] for key in ED_operators]
    print(f'total l2 distance between ED and MPS expectations = {np.linalg.norm(expectation_difference)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hamiltonian_filename')
    args = parser.parse_args()

    print('loading Hamiltonian')
    H_paulis, H_coeffs, H_params = state_simulator.loadHamiltonian(args.hamiltonian_filename)
    for i in range(len(H_paulis)):
        print(f'{H_paulis[i]}  {H_coeffs[i]}')
    
    testAgainstED(H_paulis, H_coeffs, H_params)



        


