# Modeling CRISPR gene drives for suppression of invasive rodents

<hr style=\"border:3px solid gray\"> </hr>

Files and data to accompany "Modeling CRISPR gene drives for suppression of invasive rodents".

The "Figures" folder contains high resolution verisons of the figures used in the paper.

The "Animnations" folder contains severeal animated 3-at-a-time analyses to augment the analyses done in the paper.

The "Data" folder contains data used to train and test the gaussian process models.

The "Models" folder contains pre-trained GPyTorch models. These models are not human readable.

Scripts and code in this repository:
- "rat_gene_drive_model.slim": the population model used in the paper.
- "rat_gp.py": the code for the GP models.
- "rat_plot.py": plotting functions for use with the GP models.
- "GeneDriveForInvasiveRodentSuppression.ipynb": a jupyter notebok that can load the pretrained models, which can then generate additional sensitivity analyses, two-at-a-time plots, or three-at-a-time animations. No programming experience is necessary to use this notebook, and the ploting functionality is easy to use.

___
## Requirements
Running the population model requires the SLiM evolutionary simulation framework. https://messerlab.org/slim/

Runnining the gaussian process models in a jupyter notebook has a number of requirements:
- Python 3.6 or above: https://www.python.org/downloads/. Note: PyTorch requires the 64 bit Python version of python. To install this, you may have to poke around the Python downloads section. For python 3.8.3, navigate to the bottom of this page: https://www.python.org/downloads/release/python-383/
- The following python packages, which could be installed via pip or Conda. Pip commands are as follows:
  - PyTorch: install varies from machine to machine, see https://pytorch.org/get-started/locally/ to configure the required command for your machine. If you want to accelerate the code with your GPU for potentially much faster runtimes, install NVidia's CUDA toolkit first: https://developer.nvidia.com/cuda-downloads.  
  - Jupyter notebook: install via running ``pip install jupyterlab`` in a terminal or command prompt window or by other means.
  - GPyTorch: install via ``pip install gpytorch``. Install only after installing torch.
  - SALib: install via ``pip install SALib`` or by other means.

Installing these packages will automatically install other required packages, such as numpy, pandas, and matplotlib.

The notebook can display animated 3-at-a-time heatmap plots using a javascript widget. This might not work in all browsers, in which case the plots can be generated as a movie file. This requires ffmpeg: https://ffmpeg.org/download.html.

___
## Running the notebook
After requirements are installed, the jupyter notebook can be run by opening a terminal or command prompt in the folder with these files, and running:

```
jupyter notebook GeneDriveForInvasiveRodentSuppression.ipynb
```

Instructions on using the GP models is present within the notebook.
