# Continuum model for an epithelial tissue
Calculation of flow field and stresses in a model for epithelial tissue with active mechano-chemical feedback. 

The model is described in my paper here: https://doi.org/10.1103/PhysRevLett.131.238301 

## How to use
- Input parameters and run "main.py" using "run_batch.sh". 
- "main.py" carries out the numerical solution for the coupled differential equations in the model for the actomyosin tensor, passive pressure, and velocity field.
- You can generate a movie of the simulation with "make_movie.py". The function "utils.plot_fields()" takes optional arguments - see comments in "make_movie.py"

## Dependencies and Packages
Outside of the packages in the standard python library, you will need these:
- [ffmpeg](https://ffmpeg.org/) for turning a series of plots into a movie
- [Matplotlib](https://matplotlib.org/) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- [NumPy](https://numpy.org/) is the standard package for scientific computing with Python
