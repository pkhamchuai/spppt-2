import subprocess

dataset = range(0, 6)
# dataset = [0]

for i in dataset:
    # Run the SP batch file
    subprocess.run(['python', 'run_SP.py', '--model', 'SP', 
                    '--sup', '1', '--dataset', str(i)])
