import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

seed = 42
np.random.seed(seed)


def avg_estimator(signal, norm):
    # Keys:
    k = np.sin(k_2pif * t.T)
    # Attention Matrix:
    s = (np.dot(signal.T, k.T) * norm)[0]
    s -= max(s)
    s = np.exp(s)
    s /= np.sum(s)
    return np.dot(s, f_matrix)[0]


# Hyperparameters:
df = 0.1
iterations = 10 ** 4
no_sins = 1

# Time Axis:
fs = 1000
N = 1000
n = np.linspace(1, N, num=N)
t = np.reshape(n, (n.size, 1)) / fs  # (N,) --> (N, 1)

# normalization:
start = -0.75
stop = 0
step = 5e-4
alphas = np.arange(start=start, stop=stop + step, step=step)

# Pick random frequencies in [fmin + e, fmax - e]:
e = 1
fmin = 0 + e
fmax = fs // 2 - e
ground = np.random.uniform(low=fmin, high=fmax, size=(len(alphas), iterations))

# Frequency matrix:
f_matrix = np.arange(start=fmin, stop=fmax, step=df)
f_matrix = np.reshape(f_matrix, (f_matrix.size, 1))  # (N,) --> (N, 1)

# Key matrix 2*pi*f
k_2pif = 2 * np.pi * f_matrix
# Sinusoid's 2*pi*t:
s_2pit = 2 * np.pi * t

# Store predictions:
pred_avg = np.zeros(shape=(len(alphas), iterations))

# Store average MAE for each alpha:
amae = np.zeros(shape=(len(alphas)))

# For each alpha:
for alpha in tqdm(range(len(alphas)), desc="Alphas"):

    for i in range(iterations):

        # Create a sinusoid with a random frequency:
        x = np.sin(ground[alpha, i] * s_2pit)

        # Predict its frequency:
        pred_avg[alpha, i] = avg_estimator(signal=x, norm=N ** alphas[alpha])

    # Store average MAE for this alpha:
    amae[alpha] = np.sum(np.abs(ground[alpha] - pred_avg[alpha])) / iterations

path = "np"
Path(path).mkdir(parents=True, exist_ok=True)
np.save(file=f"{path}/ground.npy", arr=ground)
np.save(file=f"{path}/avg.npy", arr=pred_avg)
np.save(file=f"{path}/alphas.npy", arr=alphas)
np.save(file=f"{path}/amae.npy", arr=amae)

plt.figure()
plt.plot(alphas, amae)
plt.title("alpha VS AMAE")
plt.xlabel("alpha")
plt.ylabel("AMAE")
plt.xlim([alphas[0], alphas[-1]])
plt.ylim([min(amae), max(amae)])
plt.show()
plt.close()
