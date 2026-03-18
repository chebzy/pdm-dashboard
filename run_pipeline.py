import subprocess
import sys
import time

steps = [
    "simulate_dataset.py",
    "train_rul_model.py",
    "generate_latest_snapshot.py",
]

start = time.time()

for step in steps:
    print(f"\nRunning {step}...")
    subprocess.run([sys.executable, step], check=True)

elapsed = time.time() - start
print(f"\nPipeline completed successfully in {elapsed:.1f} seconds.")