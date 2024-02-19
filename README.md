# hamiltonian-learning

./hamiltonians contains some classes of hamiltonians
./hamiltonian_learning.py contains the learnHamiltonianFromThermalState method, which is the main algorithm
./setup_thermal.yml are the settings that thermal_tester.py is to be run with
./simulation.py contains some methods used to generate the thermal states that the tester uses
./thermal_tester.py is the script that generates (or loads) a thermal state, runs the algorithm on it, analyzes the results, and prints, plots, and saves the results of the analysis
./utils.py contains some methods used by the other pythons scripts

How to use:
1. Modify setup_thermal.yml to the desired parameters
2. run "python thermal_tester.py setup_thermal.yml" (for convenience, some parameters can be passed from the command line without modifying setup_thermal.py. See thermal_tester.py for details)
