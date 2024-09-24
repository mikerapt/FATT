import librosa
import numpy as np
from os import listdir
from os.path import isfile
from os.path import join as join
import matplotlib.pyplot as plt

LOG_BAR = 64 * "-"


def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


def load_vowels(path=None, sec=0.06, sr=8000):
    """Load vowels from directory `path` found in the CWD"""
    vowels, names = [], []
    for name in listdir(path):
        if isfile(join(path, name)):
            names.append(name)
            vowel, _ = librosa.load(join(path, name), sr=sr)
            x = vowel[sr: sr + int(sr * sec)]
            x /= max(abs(x))
            x -= np.mean(x)
            vowels.append(x)
    return np.asarray(vowels), np.asarray(names)


def plotter(time, original, qifft, fattls, show=True, save_to=None, name=None):
    """Plots original vs reconstructions and residuals in time domain."""

    plt.figure(figsize=(10, 15))

    # Original vs. FATTLS reconstruction:
    plt.subplot(3, 1, 1)  # (rows, columns, subplot number)
    plt.plot(time, original, label="Original")
    plt.plot(time, fattls, label="FATTLS")
    plt.xlim([min(time), max(time)])
    plt.ylim([np.min([original, fattls]), np.max([original, fattls])])
    plt.xlabel("Time (sec)")
    plt.grid()
    plt.legend(loc="upper left")
    plt.title("Original vs. FATT-LS reconstruction")

    # Original vs. QIFFT reconstruction:
    plt.subplot(3, 1, 2)
    plt.plot(time, original, label="Original")
    plt.plot(time, qifft, label="QIFFT")
    plt.xlim([min(time), max(time)])
    plt.ylim([np.min([original, qifft]), np.max([original, qifft])])
    plt.xlabel("Time (sec)")
    plt.grid()
    plt.legend(loc="upper left")
    plt.title("Original vs. QIFFT reconstruction")

    # FATTLS residual vs. QIFFT residual:
    res_qifft = original - qifft
    res_fattls = original - fattls
    plt.subplot(3, 1, 3)
    plt.plot(time, res_qifft, label="QIFFT residual")
    plt.plot(time, res_fattls, label="FATT-LS residual")
    plt.xlim([min(time), max(time)])
    plt.ylim(
        [np.min([res_qifft, res_fattls]), np.max([res_qifft, res_fattls])]
    )
    plt.xlabel("Time (sec)")
    plt.grid()
    plt.legend(loc="upper left")
    plt.title("Residuals")

    plt.tight_layout()
    if save_to:
        plt.savefig(f"{join(save_to, name)}.pdf")
    if show:
        plt.show()
    plt.close()
    return
