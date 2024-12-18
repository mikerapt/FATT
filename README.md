# Fourier Attention (FATT) - Supplementary Code

## Overview

This repository contains the official implementation of FATT with the Least Squares extension (FATT-LS), as proposed in the paper titled "Fourier Attention: The Attention Mechanism as a Frequency Analyzer". The goal of this project is to explore the use of the attention mechanism for frequency analysis in speech and signal processing tasks, offering an alternative to traditional methods like the Quadratically Interpolated Fast Fourier Transform (QIFFT). This approach not only provides a novel method for frequency analysis but also lays the groundwork for a potentially new type of trainable layer, which could prove useful in more complex tasks, such as Text-To-Speech (TTS) synthesis. As an example, we compare the reconstruction of 10 recorded male and female speech vowel signals using both the FATT-LS and QIFFT methods.

## Dependencies

In a python 3.10.13 environment install dependencies by running:
```bash
pip install -r requirements.txt
```

## File Descriptions

- [`qifft.py`](./qifft.py) Contains the implementation of the QIFFT algorithm, which is used for comparison against FATT-LS.
- [`fattls.py`](./fattls.py) Implements FATT-LS, the proposed method.
- [`utils.py`](./utils.py) Contains utility functions used throughout the code, such as data loading, preprocessing, and plotting.
- [`vowels.py`](./vowels.py) Includes an example experimental setup for vowel analysis, comparing the performance of FATT and QIFFT on vowel frames.

## Usage

To run the vowel analysis with default settings and compare FATT-LS with QIFFT, execute:

```bash
python vowels.py
```
A folder will be generated with the results that compare the reconstructed and residual signals from both algorithms.

To see available options (e.g., sampling rate, number of sinusoids, input length, etc.), run:
```bash
python vowels.py --help
```

## Citation

If you want to use this code in your research or project, please cite the associated paper. For more details, you can view the full paper [here](https://www.isca-archive.org/iberspeech_2024/raptakis24_iberspeech.pdf).
