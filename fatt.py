from scipy.special import softmax
import numpy as np


class FATTLS:
    def __init__(self, fs, time, df, f_min, f_max, no_sins, a, mode):
        # Hyperparameters:
        self.fs = fs
        self.t = np.reshape(a=time, newshape=(len(time), 1))
        self.pi2t = 2 * np.pi * self.t
        self.df = df
        self.no_sins = no_sins
        self.norm = len(time)**a
        # FATT matrices:
        self.f_matrix = np.arange(start=f_min, stop=f_max, step=self.df)
        self.f_matrix = np.reshape(a=self.f_matrix, newshape=(self.f_matrix.size, 1))
        self.values = self.f_matrix
        self.keys = np.concatenate(
            (np.sin(self.pi2t.T * self.f_matrix),
             np.cos(self.pi2t.T * self.f_matrix)),
            axis=0
        )
        # FATT estimator (output):
        if mode == "avg" or mode == "average":
            self.output = self.average
        elif mode == "med" or mode == "median:":
            self.output = self.median
        else:
            self.output = self.maximum
        # Least Squares matrix:
        self.m = np.empty((len(time), 0))

    def average(self, s):
        return np.matmul(s, self.values)[0]

    def maximum(self, s):
        return self.f_matrix[np.argmax(s)][0]

    def median(self, s):
        cumarea = np.cumsum(s)
        med_idx = np.argmin(np.abs(cumarea - cumarea[-1] / 2))
        return self.f_matrix[med_idx][0]

    def reconstruct(self, freqs, amps):
        """Reconstruct signal based on the Fourier attention sinusoidal parameters."""
        return np.sum(
            amps[::2] * np.sin(self.pi2t * freqs) + amps[1::2] * np.cos(self.pi2t * freqs), axis=1
        ).reshape(-1, 1)

    def least_squares(self, signal, freq):
        """Returns a_hat and b_hat of Fourier attention."""
        # Update least squares matrix:
        self.m = np.concatenate(
            (self.m,
             np.sin(self.pi2t * freq),
             np.cos(self.pi2t * freq)),
            axis=1
        )
        # Return the least square minimizers alpha, beta:
        return np.matmul(np.matmul(np.linalg.pinv(np.matmul(self.m.T, self.m)), self.m.T), signal)[:, 0]

    def fatt(self, signal):
        """Estimate a frequency based on the Fourier attention mechanism."""

        # Querries:
        # q = signal.T

        # Q * K^T * n^a:
        t = np.matmul(signal.T, self.keys.T) * self.norm
        t = np.sqrt((t[0, 0:self.f_matrix.shape[0]] ** 2) + (t[0, self.f_matrix.shape[0]:] ** 2))

        # Attention Matrix:
        # s = softmax(t - max(t))

        # Output:
        return self.output(softmax(t - max(t)))

    def estimate(self, signal):
        """Estimate FATT sinusoidal parameters."""

        # Initialize:
        sig = np.reshape(a=signal, newshape=(len(signal), 1))
        estimated = np.zeros(shape=(len(sig), 1))
        freqs = np.zeros(self.no_sins)
        amps = np.zeros(self.no_sins*2)
        residual = sig

        # Main loop:
        for i in range(self.no_sins):

            # Estimate frequency via FATT:
            freqs[i] = self.fatt(signal=residual)

            # Estimate alpha, beta amplitudes via least squares:
            amps[:2*(i+1)] = self.least_squares(signal=sig, freq=freqs[i])

            # Get reconstruction:
            estimated = self.reconstruct(freqs=freqs, amps=amps)

            # Subtract reconstruction from original and repeat:
            residual = sig - estimated

        # Re-initialize least squares matrix:
        self.m = np.empty((len(self.t), 0))

        return freqs, amps, estimated[:, 0]
