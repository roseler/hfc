import time
import psutil
import pandas as pd

def monitor_and_run(fn, *args, **kwargs):
    """Run a function and record performance metrics: time, memory usage."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    end_time = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    execution_time = round(end_time - start_time, 3)
    memory_used = round(mem_after - mem_before, 2)

    stats = {
        "Execution Time (s)": execution_time,
        "Memory Increase (MB)": memory_used
    }

    return result, stats

# Example monitoring stubs (dummy functions to demonstrate structure)
def dummy_pfe_pipeline(X):
    time.sleep(2)  # Simulate processing
    return pd.DataFrame(X)  # Return as-is for demo

def dummy_hfc_pipeline(X, y):
    time.sleep(1.2)  # Simulate lighter processing
    return pd.DataFrame(X), {}  # Return dummy output

# Simulated input data
sample_X = pd.DataFrame({
    "f1": range(1000),
    "f2": range(1000, 2000)
})
sample_y = pd.Series([0, 1] * 500)

# Run monitor tests
pfe_result, pfe_stats = monitor_and_run(dummy_pfe_pipeline, sample_X)
hfc_result, hfc_stats = monitor_and_run(dummy_hfc_pipeline, sample_X, sample_y)

# Output results
pfe_stats, hfc_stats
