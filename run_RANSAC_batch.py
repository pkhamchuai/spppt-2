import subprocess

# dataset = [0, 1, 2, 3]
# dataset = range(1, 6)
dataset = [6]
plot_ = 1

for i in dataset:
    # Run the SP batch file
    subprocess.run(['python', 'run_RANSAC.py', '--model', 'LMEDS', 
                    '--sup', '0', '--dataset', str(i), '--plot', str(plot_)])

    subprocess.run(['python', 'run_RANSAC.py', '--model', 'RANSAC', 
                    '--sup', '0', '--dataset', str(i), '--plot', str(plot_)])
