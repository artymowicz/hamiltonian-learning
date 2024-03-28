# Hamiltonian learning
Implementation of the Hamiltonian learning algorithm described in the paper _Efficient Hamiltonian reconstruction from equilibrium states_ (https://arxiv.org/abs/2403.18061). The file `hamiltonian_learning.py` contains the function `learnHamiltonianFromThermalState` which implements the algorithm. The file `tester.py` simulates a Hamiltonian reconstruction problem with measurement noise.

### Dependencies
`hamiltonian_learning.py` requires:
- python 3 with numpy (https://numpy.org/) and scipy (https://scipy.org/)
- pyYAML (https://pyyaml.org/wiki/PyYAMLDocumentation)
- MOSEK solver (https://www.mosek.com)

Additionally, `tester.py` requires:
- TenPy (https://tenpy.readthedocs.io/)
- h5py (https://docs.h5py.org/en/stable/)
- tqdm (https://tqdm.github.io/)

## Documentation for `hamiltonian_learning.py`
The `learnHamiltonianFromThermalState` function calls MOSEK on the dual problem and recovers the solutions as the dual variables of the constraints.

### Interface for `learnHamiltonianFromThermalState` function:
| Input parameter                                             |description                                                                |
|-------------------------------------------------------------|----------------------------------------|
| `int r`                                                     | number of perturbing operators          |
| `int s`                                                     | number of variational hamiltonian terms            |
| `ndarray[float] h_terms_exp`                                | length-s vector of floats which contains the expectations of the hamiltonian terms, ie. omega(h_alpha) for alpha = 1,...,s|
| `ndarray[complex] J`                                        | r x r complex matrix satisfying b_i* = sum_j J_ij b_j (where b_i are the perturbing operators)|
| `ndarray[complex] C`                                        | r x r complex matrix given by C_ij = omega(b_i* b_j)|
| `ndarray[int] F_indices`  <br>  `ndarray[complex] F_values` | sparse representation (in COO format) of the tensor with dimensions (r,r,s) given by F_ijk = omega(b_i*[h_k, b_j])|
| `float epsilon_W`                                           | eigenvalue threshold for computing the approximate kernel of the matrix W|
| `int printing_level`                                        | how much to print (0 means no output at all and 2 means detailed output)|


| Output parameter                        |description                                                                  |
|-----------------------------------------|-----------------------------------------------------------------------------|
|`ndarray[float] hamiltonian_coefficients`| coefficients of the recovered Hamiltonian, in the same order as h_terms_exp |
|	`float T`                               | recovered temperature                                                       |
|	`float mu`                              | regularization parameter mu                                                 |
|	`int q`                                 | number of eigenvectors of W matrix that were used in the convex optimization|

## Documentation for `tester.py`
`tester.py` is a script that generates a thermal state, runs the algorithm on it and reports the results. MPS representation of the state and its expectation values are cached after being computed, and `tester.py` will use cached state/expectations if available instead of recomputing them. If the run is successful, a directory `./runs/[current date and time]` is created where the results are saved. 

**How to use:**
1. Save the desired Hamiltonian as a .yml file in the directory `./hamiltonians/` (open one of the existing .yml files to see the required format)
2. Modify `setup.yml` to the desired parameters
3. Run `python tester.py setup.yml` (for convenience, some parameters can be passed from the command line without modifying setup.yml . See tester.py for details)

## File structure

The repo contains:

- `./hamiltonian_learning.py` 

- `./tester.py` 

- `./hamiltonians/` is where tester.py looks for hamiltonians (as .yml files) 

- `./setup.yml` are the settings that tester.py is to be run with

- `./simulation.py` contains some methods used to generate the thermal states that tester.py uses

- `./utils.py` contains some methods used by the other programs

Additionally, after the first successful run, the following directories are created:

- `./caches/` is where cached thermal states (in purified MPS form) and expectation values are stored

- `./runs/` is where results of tester.py are saved


