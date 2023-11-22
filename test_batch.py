import subprocess

dataset = [0, 1, 2, 3]
for i in dataset:
    # Run the test file
    # subprocess.run(['python', 'test.py', '--model', 'SP_AffineNet3', '--sup', '1', '--dataset', str(i), 
    #                 '--model_path', 'SP_AffineNet3_11100_0.0001_0_10_1_20231109-132956.pth'])

    # .venv/bin/python test_Rep.py --model DHR --model_path DHR_11100_0.001_0_20_1_20231030-165803.pth
    subprocess.run(['.venv/bin/python', 'test_Rep.py', '--model', 'DHR', '--dataset', str(i),
                    '--model_path', 'DHR_11100_0.001_0_20_1_20231030-165803.pth'])
