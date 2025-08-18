This repository contains the code to load and test the pre-trained generative models used in the publication *Distilling Hydrological and Land Surface Model Parameters from Physio-Geographical Properties Using Text-Generating AI*. Before running the code the pretrained models need to be downloaded and the corresponding python environment needs set up. Both steps are explained below.

## Download pre-trained models

Download the pre-trained generative models from <https://doi.org/10.5281/zenodo.16895098>

## Environment Setup

To run this project, first create a Python environment named GenAI_VAE. If you are using conda, run:

```bash
conda create -n GenAI_VAE python=3.10
conda activate GenAI_VAE
```

If you prefer venv, run:

```bash
python -m venv GenAI_VAE
source GenAI_VAE/bin/activate   # On Linux/Mac
GenAI_VAE\Scripts\activate      # On Windows
```

Once the environment is active, install the dependencies:

```bash
pip install -r requirements.txt
```

Finally, verify the installation by running:

```bash
python -c "import torch, numpy, pandas; print('Torch version:', torch.__version__)"
```

You should see Torch version: 2.4.1.

## Run genAI test
Use the main.py file to load a specific VAE-decoder model and generate some random equations from it.
