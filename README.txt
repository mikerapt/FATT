Fourier Attention Mechanism - Supplementary Code

1) Overview
This repository contains the official implementation of the Fourier Attention (FATT) mechanism with the least squares addition (FATT-LS) as proposed in the paper titled "Fourier Attention: The Attention Mechanism as a Frequency Analyzer". The goal of the project is to explore the use of the attention mechanism for frequency analysis in speech and signal processing tasks, offering an alternative to more traditional methods like the Quadratically Interpolated FFT (QIFFT). As a usage example, we compare the reconstruction of 10 recorded male and female speech vowel signals using both methods.

2) Dependencies
To run the provided scripts, you will need to install the following Python packages:

conda install numpy
conda install matplotlib
pip install librosa
pip install tqdm

3) File Descriptions
qifft.py: This file contains the implementation of the QIFFT algorithm, which is used for comparison against FATT-LS.
fattls.py: This script implements FATT-LS, which iteratively estimates sinusoidal parameters from input signals.
utils.py: Contains utility functions used throughout the code, such as handling signal processing and data loading.
vowels.py: This script includes the experimental setup for vowel analysis, which compares the performance of the FATT and QIFFT methods on recorded vowels.

4) Usage
To run the vowel analysis with default settings and compare FATT-LS with QIFFT, execute:

python vowels.py

The results will include sinusoidal parameter estimations, root mean square error (RMSE), and reconstructed signals from both algorithms.

5) Citation
If you use this code in your research, please cite the following paper:

@inproceedings{anonymous2024fourierattention,
  title={Fourier Attention: The Attention Mechanism as a Frequency Analyzer},
  author={Anonymous},
  booktitle={IberSPEECH 2024},
  year={2024}
}

For more details, see the full paper here.
