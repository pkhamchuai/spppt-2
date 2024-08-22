import subprocess

dataset = [12]
# dataset = [3, 4, 5, 6, 7, 8, 9, 11]
# dataset = [11, 9, 8, 7, 6, 5, 4, 3]
# method = ['perspective']
method = ['affine']

for i in dataset:
    # Run the SP batch file
    # subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
    #                 '--dataset', str(i), '--plot', '1',
    #                 '--method', 'LMEDS',
    #                 ])
    # subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
    #                 '--dataset', str(i), '--plot', '2',
    #                 '--method', 'LMEDS',
    #                 ])

    # subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
    #                 '--dataset', str(i), '--plot', '1',
    #                 '--method', 'RANSAC',
    #                 ])
    # subprocess.run(['python', 'run_SP.py', '--model', 'SP', '--sup', '0', 
    #                 '--dataset', str(i), '--plot', '2',
    #                 '--method', 'RANSAC',
    #                 ])

    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '1',
    #                 '--method1', 'BFMatcher', '--method2', 'LMEDS',
    #                 ])
    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '2',
    #                 '--method1', 'BFMatcher', '--method2', 'LMEDS',
    #                 ])

    # subprocess.run(['python', 'run_SIFT_test.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '2',
    #                 '--method1', 'BFMatcher', '--method2', 'RANSAC',
    #                 '--method3', 'perspective', '--num_epochs', '2000',
    #                 ])
    
    # subprocess.run(['python', 'run_SIFT.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '2',
    #                 '--method1', 'BFMatcher', '--method2', 'RANSAC',
    #                 ])

    # subprocess.run(['python', 'run_SIFT_linearEQ_3pairs.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '1',
    #                 '--method1', 'BFMatcher'])

    # subprocess.run(['python', 'run_SIFT_linearEQ_3pairs.py', '--model', 'SIFT', 
    #                 '--sup', '0', '--dataset', str(i), '--plot', '2',
    #                 '--method1', 'BFMatcher'])
    
    subprocess.run(['python', 'run_elastix.py', 
                    '--dataset', str(i), '--plot', '1',
                    '--method', str(method[0]), '--num_iter', '0',
                    ])
    subprocess.run(['python', 'run_elastix.py', 
                    '--dataset', str(i), '--plot', '2',
                    '--method', str(method[0]), '--num_iter', '0',
                    ])
    
    # subprocess.run(['python', 'run_elastix.py', 
    #                 '--dataset', str(i), '--plot', '1',
    #                 '--method', str(method[0]), '--num_iter', '1000',
    #                 ])
    # subprocess.run(['python', 'run_elastix.py', 
    #                 '--dataset', str(i), '--plot', '2',
    #                 '--method', str(method[0]), '--num_iter', '1000',
    #                 ])
    
    # subprocess.run(['python', 'run_elastix_test.py', 
    #                 '--dataset', str(i), '--plot', '1',
    #                 '--method', str(method), '--num_iter', '512',
    #                 ])
    # subprocess.run(['python', 'run_elastix_test.py', 
    #                 '--dataset', str(i), '--plot', '2',
    #                 '--method', str(method), '--num_iter', '512',
    #                 ])