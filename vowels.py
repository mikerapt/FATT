# lib imports:
import numpy as np
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser
from datetime import datetime
# local imports:
from fatt import FATTLS
from qifft import QIFFT
from utils import load_vowels, plotter, rmse


def parse_args():
    parser = ArgumentParser()

    # Save figures:
    parser.add_argument("--save", default=True)

    # Vowel frames' hyperparameters:
    parser.add_argument("--sec", default=0.06)
    parser.add_argument("--fs", default=8000)

    # Sinusoidal parameter estimation hyperparameters:
    parser.add_argument("--df", default=0.1)
    parser.add_argument("--a", default=-0.5)
    parser.add_argument("--sins", default=30)
    parser.add_argument("--fmin", default=0)
    parser.add_argument("--fmax", default=4000)
    parser.add_argument("--mode", default="max")

    return parser.parse_args()


def main():
    # Get arguments:
    args = parse_args()
    save, sec, fs = str(args.save), float(args.sec), int(args.fs)
    df, a, no_sins = float(args.df), float(args.a), int(args.sins)
    fmin, fmax, mode = float(args.fmin), float(args.fmax), str(args.mode)
    if save:
        path = Path(f"results-{datetime.today().strftime('%d-%m-%Y-%H.%M')}")
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        path = None
    del args

    # Get vowel signal frames:
    male_vowels, male_names = load_vowels(path='male', sec=sec, sr=fs)
    female_vowels, female_names = load_vowels(path='female', sec=sec, sr=fs)

    vowels = np.concatenate((male_vowels, female_vowels), axis=0)
    male_names = np.char.replace(male_names, '.wav', '-male')
    female_names = np.char.replace(female_names, '.wav', '-female')
    names = np.concatenate((male_names, female_names), axis=0)

    no_vowels = vowels.shape[0]
    len_vowels = len(vowels[0])

    # time axis definition:
    n = np.linspace(0, len_vowels - 1, num=len_vowels)
    t = np.reshape(n, (n.size, 1)) / fs

    # For same resolution: NFFT = 2 * len(f_matrix):
    nfft = 2 * len(np.arange(start=fmin, stop=fmax, step=df))

    # Keep sinusoidal parameters:
    fatt_freqs = np.zeros(shape=(no_vowels, no_sins))
    fatt_amps = np.zeros(shape=(no_vowels, no_sins * 2))
    qifft_freqs = np.zeros(shape=(no_vowels, no_sins))
    qifft_amps = np.zeros(shape=(no_vowels, no_sins))
    qifft_phases = np.zeros(shape=(no_vowels, no_sins))
    # Keep reconstructions:
    qifft_recons = np.zeros(shape=vowels.shape)
    fatt_recons = np.zeros(shape=vowels.shape)
    # Keep reconstruction error:
    fatt_rmse = np.zeros(shape=no_vowels)
    qifft_rmse = np.zeros(shape=no_vowels)
    # Keep time:
    fatt_time = np.zeros(shape=no_vowels)
    qifft_time = np.zeros(shape=no_vowels)

    # Initialize models:
    qifft_model = QIFFT(
        fs=fs,
        time=t,
        nfft=nfft,
        win=np.hanning,
        no_sins=no_sins
    )

    fattls_model = FATTLS(
        fs=fs,
        time=t,
        df=df,
        f_min=fmin,
        f_max=fmax,
        no_sins=no_sins,
        a=a,
        mode=mode
    )

    bar = 64 * '-'

    # Main loop:
    for i, vowel in enumerate(vowels):

        print(f"{bar}\nvowel = {i + 1}/{no_vowels} ({names[i]})")

        # Sinusoidal estimation with FATT + Least Squares (LS):
        tic = perf_counter()
        fatt_freqs[i], fatt_amps[i], fatt_recons[i] = fattls_model.estimate(signal=vowel)
        fatt_time[i] = perf_counter() - tic
        fatt_rmse[i] = rmse(true=vowel, pred=fatt_recons[i])
        print(f"FATTLS: {(fatt_time[i]) * 1000:.2f} ms | RMSE {fatt_rmse[i]:.3f}")

        # Sinusoidal estimation with QIFFT:
        tic = perf_counter()
        f, a, p, qifft_recons[i] = qifft_model.estimate(signal=vowel)
        qifft_freqs[i, :len(f)], qifft_amps[i, :len(a)], qifft_phases[i, :len(p)] = f, a, p
        qifft_time[i] = perf_counter() - tic
        qifft_rmse[i] = rmse(true=vowel, pred=qifft_recons[i])
        print(f"QIFFT:  {(qifft_time[i]) * 1000:.2f} ms   | RMSE {qifft_rmse[i]:.3f}")

        # Visualize reconstructions & residuals:
        plotter(
            time=t,
            original=vowel,
            qifft=qifft_recons[i],
            fattls=fatt_recons[i],
            show=False,
            save=save,
            path=path,
            name=f"{names[i]}_{i}"
        )

    print(
        f"{bar}\nFinished:"
        f"\n Average FATTLS time: {np.mean(fatt_time) * 1000:.2f} ms"
        f"\n Average FATTLS RMSE: {np.mean(fatt_rmse):.3f}"
        f"\n Average QIFFT  time: {np.mean(qifft_time) * 1000:.2f} ms"
        f"\n Average QIFFT  RMSE: {np.mean(qifft_rmse):.3f}"
        f"\n{bar}"
    )


if __name__ == "__main__":
    main()
