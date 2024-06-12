import subprocess

# dataset = range(0, 6)
dataset = [7]

for i in dataset:
    # Run the SP batch file
    subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
                    '--dataset', str(i), '--plot', '1',
                    '--method', 'LMEDS',
                    ])
    subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
                    '--dataset', str(i), '--plot', '2',
                    '--method', 'LMEDS',
                    ])

    subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
                    '--dataset', str(i), '--plot', '1',
                    '--method', 'RANSAC',
                    ])
    subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
                    '--dataset', str(i), '--plot', '2',
                    '--method', 'RANSAC',
                    ])

    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '1',
    #                 '--method1', 'BFMatcher', '--method2', 'LMEDS',
    #                 ])
    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '2',
    #                 '--method1', 'BFMatcher', '--method2', 'LMEDS',
    #                 ])

    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '1',
    #                 '--method1', 'BFMatcher', '--method2', 'RANSAC',
    #                 ])
    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '2',
    #                 '--method1', 'BFMatcher', '--method2', 'RANSAC',
    #                 ])
