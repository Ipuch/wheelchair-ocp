# wheelchair-ocp
Optimal control for wheelchair propulsion.

1. Clone the repository
```
git clone https://github.com/<your-org-or-user>/wheelchair-ocp.git
cd wheelchair-ocp
```

2. Initialize submodules

Since you added bioptim as a submodule:
```
git submodule update --init --recursive
```

This will pull the code into external/bioptim.

3. Create the Conda environment

Use the environment.yaml file (below) so that others can easily reproduce your setup:
```
conda env create -f environment.yaml
conda activate mwc
```

4. Update environment (if already created)

If you already have mwc, just run:
```
conda env update -f environment.yaml --prune
```

📄 environment.yaml
```
name: mwc
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - biorbd
  - pyorerun
  - python-graphviz
  - numpy
  - pandas
  - scipy
  - trimesh
  - matplotlib
  - ipython
  - pip
  - pip:
      # Add pip-only dependencies here if needed
      # - some-package      
```
