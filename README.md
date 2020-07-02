# Prescient
Code base for Prescient production cost model / scenario generation / prediction interval capabilities.

## Getting started

### Requirements
* Python 3.7 or later
* A mixed-integer linear programming (MILP) solver
  * Open source: CBC, GLPK, SCIP, ...
  * Commercial: Gurobi, CPLEX, ...

### Installation
1. Clone or download/extract the Prescient repository.
2. The root of the repository will contain a `setup.py` file. In the command prompt or terminal, change your working directory to the root of the repository. Execute the following command:
```
python setup.py develop
```
This will install Prescient as a Python module while obtaining necessary dependencies. It will also define some new commands in your operating system shell. After installation, your shell should recognize the following command:

```
D:\workspace\prescient>runner.py
You must list the file with the program configurations
after the program name
Usage: runner.py config_file
```

The installation of Prescient has added the `runner.py` command which is used to execute Prescient programs from the command line.

#### Solvers
We additionally recommend that users install the open source CBC MIP solver. The specific mechanics of installing CBC are platform-specific. When using Anaconda on Linux and Mac platforms, this can be accomplished simply by:

```
conda install -c conda-forge coincbc
```

The COIN-OR organization - who developers CBC - also provides pre-built binaries for a full range of platforms on https://bintray.com/coin-or/download.

### Testing Prescient
Prescient is packaged with some utility to test and demonstrate its functionality. Here we will walk through the simulation of an [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) case.

#### Additional requirements
* Git

From the command line, navigate to the following directory (relative to the repository root):

```
prescient/downloaders/
```

In that directory, there is a file named `rts_gmlc.py`. Run it:

```
python rts_gmlc.py
```

This script clones the repository of the [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) dataset and formats it for Prescient input. Once the process is complete, the location of the downloaded and processed files will be printed to the command prompt. By default, they will be in:

```
downloads/rts_gmlc/
```

Navigate to this directory. There will be a number of text files in there. These text files contain invocations and options for executing Prescient programs. For example, the `populate_with_network_deterministic.txt` file is used to run the Prescient populator for generating scenarios for one week of data using the RTS-GMLC data. These text files are provided to the `runner.py` command. Start by running the populator for one week of data.

```
runner.py populate_with_network_deterministic.txt
```

You can follow its progress through console output. Once complete, a new folder in your working directory `deterministic_with_network_scenarios` will appear. The folders within contain the scenarios generated. Now that you have scenarios, you can run the simulator for them. The instructions for running the simulator are in the `simulate_with_network_deterministic.txt` and you will once again provide them to the `runner.py` command:

```
runner.py simulate_with_network_deterministic.txt
```

Note: This is assuming that you have installed the solver CBC as recommended. If you are using another solver, you will need to edit the followiong options in the text file to specify those solvers:

```
--deterministic-ruc-solver=cbc
--sced-solver=cbc
```

You can watch the progress of the simulator as it is printed to the console. After a period of time, the simulation will be complete. The results will be found a in a new folder `deterministic_with_network_simulation_output`. In that directory, there will be a number of .csv files containing simulation results. An additional folder `plots` contains stack graphs for each day in the simulation, summarizing generation and demand at each time step.

### (For developers) Regression tests
A unit test for testing a populator and simulator run on the RTS-GMLC data is located [here](https://github.com/jwatsonnm/prescient/blob/master/tests/simulator_tests/test_pop_sim_rts-gmlc.py):

```
tests/simulator_tests/test_pop_sim_rts-gmlc.py
```

It downloads the RTS-GMLC data, runs the populator and simulator for one week, and compares simulation output to static results (saved in the repo) and uses a diff check with tolerances to evaluate the test. It’s designed to be ran in a fresh instance via Travis CI so you need not provide any data or parameters. 

The definitions of numeric fields are in numeric_fields.json; these define the fields in each output csv file with which to do results comparisons. The difference tolerances of those fields are similarly defined in tolerances.json but have default values if not explicitly defined.

You can run the script directly:

```
python test_pop_sim_rts-gmlc.py
```

or use the Python unit test interface:

```
python -m unittest test_pop_sim_rts-gmlc.py
```

### Developers

By contributing to this software project, you are agreeing to the following terms and conditions for your contributions:

You agree your contributions are submitted under the BSD license.
You represent you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer.
