Fourier Attention Mechanism - Supplementary Code

## 1) Overview
This repository contains the official implementation of the Fourier Attention (FATT) mechanism with the least squares addition (FATT-LS) as proposed in the paper titled "Fourier Attention: The Attention Mechanism as a Frequency Analyzer". The goal of the project is to explore the use of the attention mechanism for frequency analysis in speech and signal processing tasks, offering an alternative to more traditional methods like the Quadratically Interpolated FFT (QIFFT). As a usage example, we compare the reconstruction of 10 recorded male and female speech vowel signals using both methods.

## 2) Dependencies
To run the provided scripts, you will need to install the following Python packages:
```bash
conda install numpy
conda install matplotlib
pip install librosa
pip install tqdm
```

## 3) File Descriptions
```qifft.py``` This file contains the implementation of the QIFFT algorithm, which is used for comparison against FATT-LS.

```fattls.py``` This script implements FATT-LS, the proposed method.

```utils.py``` Contains utility functions used throughout the code, such as data loading, preprocessing, and plotting.

```vowels.py``` This script includes the experimental setup for vowel analysis, which compares the performance of the FATT and QIFFT methods on recorded vowels.

## 4) Usage
To run the vowel analysis with default settings and compare FATT-LS with QIFFT, execute:

```bash
python vowels.py
```
A folder will be generated with the results that compare the reconstructed signals from both algorithms. View the file itself for more options e.g., changing the sampling rate, number of sinusoids, input length, etc.

## 5) Citation
If you want to use this code in your research or project, please cite the associated paper. For more details, you can view the full paper [here] (link will be made available in the near future).
