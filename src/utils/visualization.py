# utils/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(results: dict, dataset_type: str, output_dir: str=None):
    # Determine output folder
    if output_dir is None:
        output_dir = os.path.join('figures', dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    # Use exactly the pipelines you ran
    methods = list(results.keys())

    # Prepare containers
    train_times = []
    test_times  = []
    mem_values  = []

    # 1) Gather train/test runtimes
    for method in methods:
        entry = results[method]
        if method == 'HFC':
            # entry is a dict of clustering variants
            tts, tss = [], []
            for cfg in entry.values():
                if isinstance(cfg, dict):
                    try:
                        tts.append(float(cfg.get('train_time', np.nan)))
                        tss.append(float(cfg.get('test_time',  np.nan)))
                    except (ValueError, TypeError):
                        continue
            train_times.append(np.nanmean(tts) if tts else 0.0)
            test_times .append(np.nanmean(tss) if tss else 0.0)
        else:
            # single pipeline entry
            try:
                train_times.append(float(entry.get('train_time', 0.0)))
                test_times .append(float(entry.get('test_time',  0.0)))
            except (ValueError, TypeError):
                train_times.append(0.0)
                test_times .append(0.0)

    # 2) Plot runtime comparison
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_times,  width, label='Train Time')
    ax.bar(x + width/2, test_times,   width, label='Test Time')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Time (s)')
    ax.set_title(f'{dataset_type}: Runtime Comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'runtime_comparison.png'))
    plt.close(fig)

    # 3) Gather train-memory usage if available
    for method in methods:
        entry = results[method]
        if method == 'HFC':
            # average memory across clusters if present
            mems = []
            for cfg in entry.values():
                if isinstance(cfg, dict):
                    try:
                        mems.append(float(cfg.get('train_memory', np.nan)))
                    except (ValueError, TypeError):
                        continue
            mem_values.append(np.nanmean(mems) if mems else 0.0)
        else:
            # single pipeline entry
            try:
                mem_values.append(float(entry.get('train_memory', 0.0)))
            except (ValueError, TypeError):
                mem_values.append(0.0)

    # 4) Plot memory comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(methods, mem_values, color='C1')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title(f'{dataset_type}: Train Memory Usage')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'memory_comparison.png'))
    plt.close(fig2)
