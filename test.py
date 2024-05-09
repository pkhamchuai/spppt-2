import subprocess
import os

# dataset = range(1, 4)
# sups = [1, 1, 1]
dataset = [1]
sups = [1]
models = ['DHR']
# models = ['Attention']
# , 'AIRNet', 'SP_AffineNet4'

# grab the model path in the folder 'trained_models'
# take only files with 'DHR_*'
# model_path = ['DHR_41100_0.001_0_50_40_20240306-144329.pth'] #, 'DHR_41100_0.001_0_50_100_20240306-153459.pth']
# list all files in the folder
files = os.listdir('trained_models/')
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
# '20240403-164754_Attention_stage4_00100_0.001_15_20_1.pth', # wo Avg 1230
#               '20240403-172755_Attention_stage4_00100_0.001_15_20_1.pth', # wo Avg 4530
# model_path = ['Attention_11100_0.001_0_10_50_20240409-053141.pth',
#               'Attention_21100_0.001_0_10_50_20240409-054404.pth']
model_path = ['DHR_31100_0.001_0_10_100_20240508-120807.pth']
print(model_path)

runs = []
learning_rate = 1e-3

# generate run commands
for dataset_, sup in zip(dataset, sups):
    for model in model_path:
        runs.append(['python', 'test_points.py', '--model', str('DHR'), '--sup', str(sup),
                     '--dataset', str(dataset_),
                    '--model_path', str(model)
                                ])
        runs.append(['python', 'test_two_ways.py', '--model', str('DHR'), '--sup', str(sup),
                    '--dataset', str(dataset_),
                    '--model_path', str(model)
                                ])
        runs.append(['python', 'test_rep1.py', '--model', str('DHR'), '--sup', str(sup),
                    '--dataset', str(dataset_),
                    '--model_path', str(model)
                                ])
        runs.append(['python', 'test_rep2.py', '--model', str('DHR'), '--sup', str(sup),
                     '--dataset', str(dataset_),
                    '--model_path', str(model)
                                ])
        
# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(f'\n{runs[i]}')
    subprocess.run(runs[i])
