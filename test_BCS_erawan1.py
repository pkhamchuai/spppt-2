import subprocess
import os

# sups = [0, 1, 1, 1, 1, 1]
# dataset = range(1, 6)
# sups = [1, 1, 1, 1, 1]
# dataset = [4, 5]
# sups = [1, 1]
# dataset = [12]
sups = [0]

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
# model_path = ['']
# model_path = 'trained_models/' + args.model_path

# DHR original
# models = 'DHRoriginal'
# model_path = ['DHRoriginal_11100_0.001_0_5_100_20240530-152108.pth', 'DHRoriginal_21100_0.001_0_5_100_20240530-152346.pth',
#             'DHRoriginal_31100_0.001_0_5_100_20240530-152825.pth', 'DHRoriginal_41100_0.001_0_5_100_20240530-153029.pth',
#             'DHRoriginal_51100_0.001_0_10_100_20240530-142556.pth']

# models = 'DHRdiff'
# model_path = ['DHR_11100_0.001_0_5_100_20240509-155916.pth', 'DHR_21100_0.001_0_5_100_20240509-160207.pth',
#             'DHR_31100_0.001_0_10_100_20240508-120807.pth', 'DHR_41100_0.001_0_5_100_20240509-133824.pth',
#             'DHR_51100_0.001_0_5_100_20240509-140837.pth']

# DHR 2x
# models = 'DHR2x'
# model_path = ['DHR2x_11100_0.001_0_10_100_20240515-130704.pth', 'DHR2x_21100_0.001_0_5_100_20240514-131741.pth',
#               'DHR2x_31100_0.001_0_5_100_20240514-132527.pth', 'DHR2x_41100_0.001_0_5_100_20240514-132814.pth',
#               'DHR2x_51100_0.001_0_5_100_20240513-112002.pth']

# Attention
# models = 'Attention'
# model_path = ['Attention_11100_0.001_0_5_100_20240513-152019.pth', 'Attention_21100_0.001_0_5_100_20240513-152334.pth',
#               'Attention_31100_0.001_0_5_100_20240513-152949.pth', 'Attention_41100_0.001_0_5_100_20240513-153219.pth',
#               'Attention_51100_0.001_0_5_100_20240513-153734.pth']
    
# Attention without pooling
# models = 'Attention_no_pooling'
# model_path = ['Attention_no_pooling_11100_0.001_0_5_100_20240514-095754.pth', 'Attention_no_pooling_21100_0.001_0_5_100_20240514-100109.pth',
#               'Attention_no_pooling_31100_0.001_0_5_100_20240514-100721.pth', 'Attention_no_pooling_41100_0.001_0_5_100_20240514-100950.pth',
#               'Attention_no_pooling_51100_0.001_0_5_100_20240514-101458.pth']

# All
# models = 'All'
# model_path = ['DHR_11100_0.001_0_5_100_20240509-155916.pth', 'DHR_21100_0.001_0_5_100_20240509-160207.pth',
#             'DHR_31100_0.001_0_10_100_20240508-120807.pth', 'DHR_41100_0.001_0_5_100_20240509-133824.pth',
#             'DHR_51100_0.001_0_5_100_20240509-140837.pth', 
#             'DHR2x_11100_0.001_0_10_100_20240515-130704.pth', 'DHR2x_21100_0.001_0_5_100_20240514-131741.pth',
#             'DHR2x_31100_0.001_0_5_100_20240514-132527.pth', 'DHR2x_41100_0.001_0_5_100_20240514-132814.pth',
#             'DHR2x_51100_0.001_0_5_100_20240513-112002.pth',
#             'Attention_11100_0.001_0_5_100_20240513-152019.pth', 'Attention_21100_0.001_0_5_100_20240513-152334.pth',
#             'Attention_31100_0.001_0_5_100_20240513-152949.pth', 'Attention_41100_0.001_0_5_100_20240513-153219.pth',
#             'Attention_51100_0.001_0_5_100_20240513-153734.pth',
#             'Attention_no_pooling_11100_0.001_0_5_100_20240514-095754.pth', 'Attention_no_pooling_21100_0.001_0_5_100_20240514-100109.pth',
#             'Attention_no_pooling_31100_0.001_0_5_100_20240514-100721.pth', 'Attention_no_pooling_41100_0.001_0_5_100_20240514-100950.pth',
#             'Attention_no_pooling_51100_0.001_0_5_100_20240514-101458.pth']

# print(model_path)

runs = []
learning_rate = 1e-3

models = 'DHRdiff'
rep = 60
dataset = [18]
plot = 1
verbose = 0

for i in range(1, 4):
    # remove some models
    # only scale1, shear1, rotation1, and mix2 are used
    model_path = ['DHR_11100_0.001_0_5_100_20240509-155916.pth',
                'DHR_21100_0.001_0_5_100_20240509-160207.pth',
                'DHR_31100_0.001_0_10_100_20240508-120807.pth', 
                'DHR_41100_0.001_0_5_100_20240509-133824.pth', 
                'DHRdiff_25_1100_0.001_0_10_50_20241121-213438.pth']
    for dataset_ in dataset:
        # runs.append(['python', 'test_BCS_2way_point.py', '--model', str(models), '--sup', str(sups[0]),
        runs.append(['python', 'test_BCS_1way_point_reverse_erawan.py', '--model', str(models), '--sup', str(sups[0]),
                        '--dataset', str(dataset_), '--beam', str(i), '--metric', str('TRE'),
                        '--model_path', str(model_path), '--plot', str(plot), '--verbose', str(verbose), '--rep', str(rep)])

    # remove some models
    # only scale1, shear1, rotation1, and mix2 are used
    model_path = ['DHR_11100_0.001_0_5_100_20240509-155916.pth',
                'DHR_21100_0.001_0_5_100_20240509-160207.pth',
                'DHR_31100_0.001_0_10_100_20240508-120807.pth', 
                'DHR_41100_0.001_0_5_100_20240509-133824.pth', 
                'DHRdiff_25_1100_0.001_0_10_50_20241121-213438.pth']
    for dataset_ in dataset:
        runs.append(['python', 'test_BCS_2way_point_erawan.py', '--model', str(models), '--sup', str(sups[0]),
        # runs.append(['python', 'test_BCS_1way_point_reverse_erawan.py', '--model', str(models), '--sup', str(sups[0]),
                        '--dataset', str(dataset_), '--beam', str(i), '--metric', str('TRE'),
                        '--model_path', str(model_path), '--plot', str(plot), '--verbose', str(verbose), '--rep', str(rep)])
# sort runs by element 1, then 11, then 7
# runs.sort(key=lambda x: x[3])

# runs.sort(key=lambda x: x[5])
for i in range(len(runs)):
    print(f'\n{runs[i]}')
    subprocess.run(runs[i])
