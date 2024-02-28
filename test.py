import subprocess
import os

dataset = [3]
sups = [1]
models = ['DHR']
# , 'AIRNet', 'SP_AffineNet4'

# grab the model path in the folder 'trained_models'
# take only files with 'DHR_*'
model_path = []
# list all files in the folder
files = os.listdir('trained_models/with_groupnorm/')
# files = os.listdir('trained_models/without_groupnorm/')

# iterate through the files
for file in files:
    # if the file starts with 'DHR_', but ont 'DHR_Rigid'
    # if file.startswith('* DHR_'):
    #     # append the file to model_path
    # print(file)
    model_path.append(file)
# sort the model_path
model_path.sort()
# print the model_path
print(model_path)

runs = []
learning_rate = 1e-3

# generate run commands
for model in model_path:
    for dataset_ in range(len(dataset)):
        # runs.append(['python', 'test_points.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(1),
        #                     '--model_path', str(os.path.join('without_groupnorm', model))
                                # ])
        runs.append(['python', 'test_rep1.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(1),
                        '--model_path', str(os.path.join('with_groupnorm', model))
                        ])
        # runs.append(['python', 'test_rep2.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(1),
        #                 '--model_path', str(os.path.join('with_groupnorm', model))
        #                 ])

# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(f'\n{runs[i]}')
    subprocess.run(runs[i])
