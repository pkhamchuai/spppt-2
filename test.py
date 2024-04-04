import subprocess
import os

dataset = range(0, 6)
sups = [1, 1, 1, 1, 1, 0]
# models = ['DHR']
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
model_path = [
              '20240403-221614_Attention_stage4_00100_0.001_15_20_1.pth', # w Avg 1230
              '20240403-231356_Attention_stage4_00100_0.001_15_20_1.pth' # w Avg 4530
              ]
print(model_path)

runs = []
learning_rate = 1e-3

# generate run commands
for model in model_path:
    for dataset_, sup in zip(dataset, sups):
        # runs.append(['python', 'test_points.py', '--model', str('Attention'), '--sup', str(sup),
        #              '--dataset', str(dataset_),
        #             '--model_path', str(model)
        #                         ])
        # runs.append(['python', 'test_rep1.py', '--model', str('DHR'), '--sup', str(1), '--dataset', str(dataset_),
        #                 '--model_path', str(os.path.join('with_groupnorm', model))
        #                 ])
        # runs.append(['python', 'test_rep1.py', '--model', str('Attention'), '--sup', str(sup),
        #              '--dataset', str(dataset_),
        #             '--model_path', str(model)
        #                         ])
        runs.append(['python', 'test_rep2.py', '--model', str('Attention'), '--sup', str(sup),
                     '--dataset', str(dataset_),
                    '--model_path', str(model)
                                ])
# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(f'\n{runs[i]}')
    subprocess.run(runs[i])
