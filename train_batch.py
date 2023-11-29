import subprocess

dataset = [1, 2, 3, 0]
sups = [1, 1, 1, 0]

for i in range(len(dataset)):
    # Run the test file
    subprocess.run(['python', 'train.py', '--model', 'DHR', '--sup', str(sups[i]), '--dataset', str(dataset[i]),
                    '--num_epochs', '5', '--loss_image', '5'])
    
    subprocess.run(['python', 'train.py', '--model', 'DHR', '--sup', str(sups[i]), '--dataset', str(dataset[i]),
                    '--num_epochs', '5', '--loss_image', '6'])
    
    subprocess.run(['python', 'train.py', '--model', 'SP_AffineNet4', '--sup', str(sups[i]), '--dataset', str(dataset[i]),
                    '--num_epochs', '5', '--loss_image', '5'])
    
    subprocess.run(['python', 'train.py', '--model', 'SP_AffineNet4', '--sup', str(sups[i]), '--dataset', str(dataset[i]),
                    '--num_epochs', '5', '--loss_image', '6'])
    
    # subprocess.run(['.venv/bin/python', 'test_Rep.py', '--model', 'SP_AffineNet4', '--sup', '1', '--dataset', str(i), 
    #                 '--model_path', 'SP_AffineNet4_31103_0.0001_10_15_1_20231114-125116.pth'])

    # .venv/bin/python test_Rep.py --model DHR --model_path DHR_11100_0.001_0_20_1_20231030-165803.pth
    # subprocess.run(['.venv/bin/python', 'test_Rep.py', '--model', 'DHR', '--dataset', str(i),
    #                 '--model_path', 'DHR_11100_0.001_0_10_1_20231031-151024.pth'])
