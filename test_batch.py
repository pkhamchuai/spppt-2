import subprocess

dataset = [1, 2, 3, 0]
sups = [1, 1, 1, 0]
models = ['DHR', 'SP_AffineNet4']
model_path = ['DHR_31106_0.0001_0_5_1_20231129-144719.pth', 'SP_AffineNet4_31106_0.0001_0_5_1_20231129-145150.pth']
runs = []
learning_rate = 5e-5

# generate run commands
for j in range(len(models)):
    # for i in range(len(dataset)):
    #     # print(f'\nTraining on dataset {dataset[i]} with sup {sups[i]}')
    #     runs.append(['python', 'train.py', '--model', str(models[j]), '--sup', str(sups[i]), '--dataset', str(dataset[i]),
    #                     '--num_epochs', '5', '--loss_image', '5', '--learning_rate', learning_rate])

    for i in range(len(dataset)):
        runs.append(['python', 'test.py', '--model', str(models[j]), '--sup', str(sups[i]), '--dataset', str(dataset[i]),
                        '--num_epochs', '5', '--loss_image', '6', '--learning_rate', str(learning_rate),
                        '--model_path', str(model_path[j])
                        ])

    # for i in range(len(dataset)):
    #     runs.append(['python', 'train_points.py', '--model', str(models[j]), '--sup', str(sups[i]), '--dataset', str(dataset[i]),
    #                     '--num_epochs', '5', '--loss_image', '5', '--learning_rate', learning_rate])

        
        # subprocess.run(['.venv/bin/python', 'test_Rep.py', '--model', 'SP_AffineNet4', '--sup', '1', '--dataset', str(i), 
        #                 '--model_path', 'SP_AffineNet4_31103_0.0001_10_15_1_20231114-125116.pth'])

        # .venv/bin/python test_Rep.py --model DHR --model_path DHR_11100_0.001_0_20_1_20231030-165803.pth
        # subprocess.run(['.venv/bin/python', 'test_Rep.py', '--model', 'DHR', '--dataset', str(i),
        #                 '--model_path', 'DHR_11100_0.001_0_10_1_20231031-151024.pth'])

# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(runs[i])
    subprocess.run(runs[i])