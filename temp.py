import subprocess
import os

dataset = range(1, 6)
sups = [1, 1, 1, 1, 1]
# dataset = [5]
# sups = [1]
# model_name = 'DHR2x'
model_name = 'Attention'
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
model_path = ['Attention_11100_0.001_0_5_100_20240513-152019.pth', 'Attention_21100_0.001_0_5_100_20240513-152334.pth',
              'Attention_31100_0.001_0_5_100_20240513-152949.pth', 'Attention_41100_0.001_0_5_100_20240513-153219.pth',
              'Attention_51100_0.001_0_5_100_20240513-153734.pth']

# model_path = ['Attention_no_pooling_11100_0.001_0_5_100_20240514-095754.pth', 'Attention_no_pooling_21100_0.001_0_5_100_20240514-100109.pth',
#               'Attention_no_pooling_31100_0.001_0_5_100_20240514-100721', 
# model_path = ['Attention_no_pooling_41100_0.001_0_5_100_20240514-100950.pth',
#               'Attention_no_pooling_51100_0.001_0_5_100_20240514-101458.pth']
              
# model_path = ['DHR_11100_0.001_0_5_100_20240509-155916.pth', 'DHR_21100_0.001_0_5_100_20240509-160207.pth',
#               'DHR_31100_0.001_0_10_100_20240508-120807.pth', 'DHR_41100_0.001_0_5_100_20240509-133824.pth',
#               'DHR_51100_0.001_0_5_100_20240509-140837.pth']
print(model_path)

runs = []
learning_rate = 1e-3

# generate run commands
for model in model_path:
    for dataset_, sup in zip(dataset, sups):
        runs.append(['python', 'test_points.py', '--model', str(model_name), '--sup', str(sup),
                     '--dataset', str(dataset_),
                    '--model_path', str(model)
                                ])
        # runs.append(['python', 'test_two_ways.py', '--model', str(model_name), '--sup', str(sup),
        #             '--dataset', str(dataset_),
        #             '--model_path', str(model)
        #                         ])
        # runs.append(['python', 'test_rep1.py', '--model', str(model_name), '--sup', str(sup),
        #             '--dataset', str(dataset_),
        #             '--model_path', str(model)
        #                         ])
        # runs.append(['python', 'test_rep2.py', '--model', str(model_name), '--sup', str(sup),
        #              '--dataset', str(dataset_),
        #             '--model_path', str(model)
        #                         ])
        # runs.append(['python', 'test_ensemble.py', '--model', str(model_name), '--sup', str(sup),
        #              '--dataset', str(dataset_),
        #             '--model_path', str(None), '--plot', '1'
        #                         ])
        
# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(f'\n{runs[i]}')
    subprocess.run(runs[i])
