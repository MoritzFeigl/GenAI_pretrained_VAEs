This repository contains the code to load and test the pre-trained generative models used in the publication *Distilling Hydrological and Land Surface Model Parameters from Physio-Geographical Properties Using Text-Generating AI*. Before running the code the pretrained models need to be downloaded and the corresponding python environment needs set up. Both steps are explained below.

## System requirements

### Software dependencies

The code is written for Python 3.10 and depends on the packages listed in `requirements.txt`:

* `torch==2.4.1`
* `numpy`
* `pandas`

The project has been tested on the following operating systems:

* Ubuntu 22.04 LTS (x86_64)
* Windows 11 (x86_64)

No specialized hardware is required; the demo runs comfortably on a standard desktop or laptop CPU. A CUDA-capable GPU is optional and not required for the provided workflow.

## Download pre-trained models and demo data

Download the pre-trained generative models and the accompanying demo dataset from <https://doi.org/10.5281/zenodo.16895098>. Unzip the archive and place the contents in the `trained_vaes/` directory at the repository root so the paths referenced in `main.py` resolve correctly.

## Installation guide

### Step-by-step instructions

1. Create and activate a Python environment named `GenAI_VAE`.
   ```bash
   conda create -n GenAI_VAE python=3.10
   conda activate GenAI_VAE
   ```
   or, using `venv`:
   ```bash
   python -m venv GenAI_VAE
   source GenAI_VAE/bin/activate   # On Linux/Mac
   GenAI_VAE\Scripts\activate      # On Windows
   ```
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify the installation by checking the PyTorch version:
   ```bash
   python -c "import torch, numpy, pandas; print('Torch version:', torch.__version__)"
   ```

On a standard desktop (4 CPU cores, 16 GB RAM) the environment creation and dependency installation typically take less than 10 minutes, depending primarily on the download speed for PyTorch.

## Demo

### Run the demo

After downloading the pretrained models, execute:

```bash
python main.py
```

The script loads one of the VAE decoder checkpoints, samples 10 latent vectors within the recommended bounds, and prints reconstructed transfer functions to the console. The demo completes in under one minute on a standard desktop CPU once the models are present locally.

