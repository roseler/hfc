import matplotlib.pyplot as plt
import pandas as pd
# Memory Usage Data
memory_data = {
    'Dataset': ['UNSW', 'UNSW', 'UNSW', 'UNSW',
                'CICIDS', 'CICIDS', 'CICIDS', 'CICIDS',
                'InSDN', 'InSDN', 'InSDN', 'InSDN'],
    'Method': ['HFC', 'PFE', 'MI', 'AE'] * 3,
    'Train Memory (MB)': [39.00, 1101.25, 89.80, 345.48,
                          23.34, 1504.41, 6.57, 82.57,
                          42.80, 1132.70, 203.40, 346.10]
}

df_memory = pd.DataFrame(memory_data)

# Plot Memory Usage
plt.figure(figsize=(10, 6))
for dataset in df_memory['Dataset'].unique():
    subset = df_memory[df_memory['Dataset'] == dataset]
    plt.plot(subset['Method'], subset['Train Memory (MB)'], marker='s', label=f'{dataset}')

plt.title('Figure Y: Train Memory Usage Comparison')
plt.xlabel('Feature Engineering Method')
plt.ylabel('Memory Usage (MB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
