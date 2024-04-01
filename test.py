import subprocess
import os

dataset = [4, 5]
sups = [1]
models = ['DHR']
# , 'AIRNet', 'SP_AffineNet4'

# grab the model path in the folder 'trained_models'
# take only files with 'DHR_*'
# model_path = ['DHR_41100_0.001_0_50_40_20240306-144329.pth'] #, 'DHR_41100_0.001_0_50_100_20240306-153459.pth']
# list all files in the folder
files = os.listdir('trained_models/with_groupnorm/')
# files = os.listdir('trained_models/without_groupnorm/')

# iterate through the files
# model_path = []
# for file in files:
#     # if the file starts with 'DHR_', but ont 'DHR_Rigid'
#     if file.startswith('DHR_5'):
#         #     # append the file to model_path
#         # print(file)
#         model_path.append(file)
# # sort the model_path
# model_path.sort()
# print the model_path
model_path = ['DHR_51100_0.001_100_120_50_20240313-170255.pth']
print(model_path)

runs = []
learning_rate = 1e-3

# generate run commands
for model in model_path:
    for dataset_ in dataset:
        runs.append(['python', 'test_two_ways.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(dataset_),
                        '--model_path', str(os.path.join('with_groupnorm', model))
                                ])
        # runs.append(['python', 'test_rep1.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(dataset_),
        #                 '--model_path', str(os.path.join('with_groupnorm', model))
        #                 ])
        # runs.append(['python', 'test_rep2.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(dataset_),
        #                 '--model_path', str(os.path.join('with_groupnorm', model))
        #                 ])
        # runs.append(['python', 'test_rep3.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(1),
        #                 '--model_path', str(os.path.join('with_groupnorm', model))
        #                 ])
# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(f'\n{runs[i]}')
    subprocess.run(runs[i])
