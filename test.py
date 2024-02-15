import subprocess
import os

dataset = [3]
sups = [1]
models = ['DHR_Attn']
# , 'AIRNet', 'SP_AffineNet4'

# grab the model path in the folder 'trained_models'
# take only files with 'DHR_*'
model_path = []
# list all files in the folder
files = os.listdir('trained_models/keep/')
# iterate through the files
for file in files:
    # if the file starts with 'DHR_', but ont 'DHR_Rigid'
    if file.startswith('DHR_Attn') and not file.startswith('DHR_Rigid'):
        # append the file to model_path
        print(file)
        model_path.append(file)

runs = []
learning_rate = 1e-3

# generate run commands
for model in range(len(models)):
    # for i in range(len(dataset)):
    #     # print(f'\nTraining on dataset {dataset[i]} with sup {sups[i]}')
    #     runs.append(['python', 'train.py', '--model', str(models[j]), '--sup', str(sups[i]), '--dataset', str(dataset[i]),
    #                     '--num_epochs', '5', '--loss_image', '5', '--learning_rate', learning_rate])

    for path in range(len(model_path)):
        for dataset_ in range(len(dataset)):
            runs.append(['python', 'test_points.py', '--model', str(models[model]), '--sup', str(1), '--dataset', str(dataset[dataset_]),
                            '--model_path', str(os.path.join('keep', model_path[path]))
                            ])
            runs.append(['python', 'test_rep1.py', '--model', str(models[model]), '--sup', str(1), '--dataset', str(dataset[dataset_]),
                            '--model_path', str(os.path.join('keep', model_path[path]))
                            ])
            runs.append(['python', 'test_rep2.py', '--model', str(models[model]), '--sup', str(1), '--dataset', str(dataset[dataset_]),
                            '--model_path', str(os.path.join('keep', model_path[path]))
                            ])
                # pass
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
    # subprocess.run(runs[i])