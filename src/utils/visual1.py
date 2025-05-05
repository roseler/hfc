import matplotlib.pyplot as plt

# Methods and datasets
methods = ['HFC', 'PFE', 'MI', 'AE']
datasets = ['UNSW', 'CICIDS', 'InSDN']

# Runtime data (seconds)
runtime = {
    'UNSW': [0.856, 0.776, 19.790, 129.698],
    'CICIDS': [0.661, 1.121, 18.319, 73.849],
    'InSDN': [3.836, 2.153, 17.598, 132.219]
}

# Memory usage data (MB)
memory = {
    'UNSW': [39.00, 1101.25, 89.80, 345.48],
    'CICIDS': [23.34, 1504.41, 6.57, 82.57],
    'InSDN': [42.80, 1132.70, 203.40, 346.10]
}

# Setup side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot Runtime
for ds in datasets:
    axs[0].plot(methods, runtime[ds], marker='o', label=ds)
axs[0].set_title("Figure X: Training Runtime Comparison")
axs[0].set_xlabel("Feature Engineering Method")
axs[0].set_ylabel("Time (seconds)")
axs[0].legend()
axs[0].grid(True)

# Plot Memory
for ds in datasets:
    axs[1].plot(methods, memory[ds], marker='s', label=ds)
axs[1].set_title("Figure 25: Training Memory Usage Comparison")
axs[1].set_xlabel("Feature Engineering Method")
axs[1].set_ylabel("Memory Usage (MB)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
