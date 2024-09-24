# lib imports:
import logging
import numpy as np
from pathlib import Path
from time import perf_counter
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# local imports:
from qifft import QIFFT
from fattls import FATTLS
from utils import load_vowels, plotter, rmse, LOG_BAR

seed = 42
np.random.seed(seed)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compare the performance of the FATT-LS and QIFFT"
            " methods for sinusoidal analysis/synthesis on recorded vowels."
            " For more details, please visit the original paper."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save", type=bool, default=True,
        help="Save figures to file."
    )
    parser.add_argument(
        "--male", default="male",
        help="Directory of male vowels found in the script's CWD.",
    )
    parser.add_argument(
        "--female", default="female",
        help="Directory of female vowels found in the script's CWD.",
    )
    # Vowel frames' hyperparameters:
    parser.add_argument(
        "--sec", type=float, default=0.06,
        help="Input signal duration in seconds."
    )
    parser.add_argument(
        "--fs", type=int, default=8000,
        help="Input signal sampling rate."
    )
    # Sinusoidal parameter estimation hyperparameters:
    parser.add_argument(
        "--df", type=float, default=0.1,
        help=(
            "The FATT frequency step: Defines how fine-grained the"
            " FATT frequency matrix is, i.e., the desired decimal"
            " point accuracy of FATT's frequency predictions."
            " Decreasing df increases memory requirements but"
            " gives more precision to FATT, and vise versa."
        )
    )
    parser.add_argument(
        "--a", type=float, default=-0.5,
        help="The FATT normalization parameter."
    )
    parser.add_argument(
        "--sins", type=int, default=30,
        help="Maximum number of sinusoids to be estimated."
    )
    parser.add_argument(
        "--fmin", type=float, default=0,
        help="Minimum detectable frequency."
    )
    parser.add_argument(
        "--fmax", type=float, default=4000,
        help="Maximum detectable frequency."
    )
    parser.add_argument(
        "--mode", default="max",
        choices=("avg", "average", "med", "median", "max", "maximum"),
        help="FATT frequency estimator choice."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if we save:
    if args.save:
        path = Path(f"results-{datetime.today().strftime('%d-%m-%Y-%H.%M')}")
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        path = None

    # Logging config, log to stdout and file (if any):
    logging_handlers = [logging.StreamHandler()]
    if path:
        logging_handlers.append(logging.FileHandler(path / "log.txt"))
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=logging_handlers
    )

    # Get vowel signal frames:
    male_vowels, male_names = load_vowels(
        path=args.male, sec=args.sec, sr=args.fs
    )
    female_vowels, female_names = load_vowels(
        path=args.female, sec=args.sec, sr=args.fs
    )
    vowels = np.concatenate((male_vowels, female_vowels), axis=0)
    male_names = np.char.replace(male_names, ".wav", "-male")
    female_names = np.char.replace(female_names, ".wav", "-female")
    names = np.concatenate((male_names, female_names), axis=0)
    no_vowels = vowels.shape[0]

    # Time axis definition:
    n = np.linspace(0, len(vowels[0]) - 1, num=len(vowels[0]))
    t = np.reshape(n, (n.size, 1)) / args.fs
    # For same resolution (fairness): NFFT = 2 * len(f_matrix):
    nfft = 2 * len(np.arange(start=args.fmin, stop=args.fmax, step=args.df))
    # Keep predicted sinusoidal parameters:
    fattls_freqs, fattls_amps = (
        np.zeros((no_vowels, size)) for size in [args.sins, args.sins * 2]
    )
    qifft_freqs, qifft_amps, qifft_phases = (
        np.zeros((no_vowels, args.sins)) for _ in range(3)
    )
    # Keep reconstructions, RMSE, and execution time:
    qifft_recons, fattls_recons = (np.zeros(vowels.shape) for _ in range(2))
    fattls_rmse, qifft_rmse, fattls_time, qifft_time = (
        np.zeros(no_vowels) for _ in range(4)
    )
    # Initialize the two sinusoidal estimators:
    qifft_model = QIFFT(
        fs=args.fs, time=t, nfft=nfft, win=np.hanning, no_sins=args.sins
    )
    fattls_model = FATTLS(
        fs=args.fs, time=t, df=args.df, f_min=args.fmin, f_max=args.fmax,
        no_sins=args.sins, a=args.a, mode=args.mode,
    )

    # Main loop:
    for i, vowel in enumerate(vowels):
        logging.info(f"{LOG_BAR}\nvowel = {i + 1}/{no_vowels} ({names[i]})")

        # Sinusoidal estimation with FATT + Least Squares (FATT-LS):
        tic = perf_counter()
        fattls_freqs[i], fattls_amps[i], fattls_recons[i] = (
            fattls_model.estimate(signal=vowel)
        )
        fattls_time[i] = perf_counter() - tic
        fattls_rmse[i] = rmse(true=vowel, pred=fattls_recons[i])
        logging.info(
            f"FATT-LS: {fattls_time[i] * 1000:<6.2f} ms | "
            f"RMSE {fattls_rmse[i]:.3f}"
        )

        # Sinusoidal estimation with QIFFT:
        tic = perf_counter()
        f, a, p, qifft_recons[i] = qifft_model.estimate(signal=vowel)
        (
            qifft_freqs[i, : len(f)],
            qifft_amps[i, : len(a)],
            qifft_phases[i, : len(p)],
        ) = (f, a, p)
        qifft_time[i] = perf_counter() - tic
        qifft_rmse[i] = rmse(true=vowel, pred=qifft_recons[i])
        logging.info(
            f"QIFFT:   {qifft_time[i] * 1000:<6.2f} ms | "
            f"RMSE {qifft_rmse[i]:.3f}"
        )

        # Visualize reconstructions & residuals:
        plotter(
            time=t, original=vowel,
            qifft=qifft_recons[i], fattls=fattls_recons[i],
            show=False, save_to=path, name=f"{names[i]}_{i}",
        )

    logging.info(
        "\n".join(
            (
                LOG_BAR,
                "Finished:",
                f" Average FATT-LS time: {np.mean(fattls_time) * 1000:.2f} ms",
                f" Average FATT-LS RMSE: {np.mean(fattls_rmse):.3f}",
                f" Average QIFFT  time: {np.mean(qifft_time) * 1000:.2f} ms",
                f" Average QIFFT  RMSE: {np.mean(qifft_rmse):.3f}",
                LOG_BAR,
            )
        )
    )


if __name__ == "__main__":
    main()
