import numpy as np
from scipy.signal import find_peaks


class QIFFT:
    """This class implements the Quadratically Interpolated
    Fast Fourier Transform (QIFFT) for sinusoidal analysis/synthesis."""
    def __init__(self, fs, time, nfft, win, no_sins):
        self.fs = fs
        self.t = np.reshape(a=time, newshape=len(time))
        self.pi2t = 2 * np.pi*self.t
        self.nfft = nfft
        self.w = win(len(time))
        self.w /= sum(self.w)
        self.no_sins = no_sins

    def reconstruct(self, freqs, amps, phases):
        """Reconstruct signal based on the
        estimated QIFFT sinusoidal parameters."""
        return np.sum(
            amps[:, np.newaxis] * np.cos(
                self.pi2t * freqs[:, np.newaxis] + phases[:, np.newaxis]
            ), axis=0
        )

    def fft(self, signal):
        """Calculate FFT and return half of the
        spectrum (signals are real) and the axis."""
        signal_fft = np.fft.fft(a=signal, n=self.nfft)
        length = len(signal_fft)
        return signal_fft[:length // 2], np.arange(length)[:length // 2]

    def estimate(self, signal):
        """Estimate the sinusoidal parameters
        of the input signal with the QIFFT method."""
        # Apply window:
        sig = signal * self.w
        # Get FFT amplitude and phase responses:
        sig_fft, freq_ax = self.fft(signal=sig)
        sig_abs_fft = np.abs(sig_fft)
        sig_phase = np.unwrap(np.angle(sig_fft))
        # Find frequency peaks:
        peaks, _ = find_peaks(sig_abs_fft, height=0)
        # Initialize:
        freqs, amps, phases = (np.zeros(len(peaks)) for _ in range(3))
        # Main loop:
        for i, peak in enumerate(peaks):
            # Estimate each sinusoid frequency and amplitude
            # based on quadratic interpolation:
            x1, x2, x3 = peak - 1, peak, peak + 1
            y1, y2, y3 = sig_abs_fft[x1], sig_abs_fft[x2], sig_abs_fft[x3]
            x0 = ((y3 - y2) * (x2 + x1) - (y2 - y1) * (x3 + x2)) /\
                 (2 * (y3 - 2 * y2 + y1))
            a = (y2 - y1) / (x2 + x1 - 2 * x0)
            y0 = y1 - a * (x1 - x0) ** 2
            # Convert to Hz (double because we work with half the spectrum):
            freqs[i], amps[i] = x0 * self.fs / self.nfft, 2 * y0
            # Estimate the phase with linear interpolation:
            if x0 < x2:
                px1, px2 = x1, x2
            else:
                px1, px2 = x2, x3
            py1, py2 = sig_phase[px1], sig_phase[px2]
            phases[i] = (py2 - py1) * (x0 - px1) / (px2 - px1) + py1
        # Sort the {amplitude,frequency,phase} triplets
        # based on amplitude in descending order:
        amps, freqs, phases = zip(
            *sorted(zip(amps, freqs, phases), reverse=True)
        )
        amps, freqs, phases = np.array(amps), np.array(freqs), np.array(phases)
        if self.no_sins < len(freqs):
            freqs, amps, phases = (
                arr[:self.no_sins] for arr in (freqs, amps, phases)
            )
        return freqs, amps, phases, self.reconstruct(freqs, amps, phases)
