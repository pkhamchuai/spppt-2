import subprocess

# dataset = [0, 1, 2, 3]
# dataset = range(1, 6)
dataset = [16]

for i in dataset:
    # Run the SP batch file
    subprocess.run(['python', 'run_RANSAC.py', '--model', 'RANSAC', 
                    '--sup', '0', '--dataset', str(i)])
