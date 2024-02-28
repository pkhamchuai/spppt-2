import subprocess

dataset = [1]
model = 'DHR'
# models = ['DHR_Attn']
# models = ['DHR', 'AIRNet', 'SP_AffineNet4']
# models = ['SP_AffineNet4', 'DHR']
sups = [1, 0]
runs = []
# learning_rate = 1e-4
# loss = [[1, 1]]
# offset = 0.01
# sigma = range(30, 100, 10)
files = ['train_img_batch.py']
# files = ['train_points_rigid_img', 'train_points_rigid_pt', 'train_img_batch.py', 'train_one_sample.py']
batch_size = [1] # 2, 3, 4, 5
lr = 1e-3
decay_rate = 0.96

# generate run commands
for file in files:
    sup = 1
    for batch in batch_size:
        runs.append(['python', str(file), '--dataset', str(1), 
            '--model', str(model), '--num_epochs', str(5), 
            '--learning_rate', str(lr), '--decay_rate', str(decay_rate),
            '--image', str(0), '--points', str(1), '--sup', str(sup),
            '--batch_size', str(batch)])  
    # for data in dataset:
    #     # if model != 'AIRNet' and sup == 0:
    #     #         pass
    #     # else:
    #     # model = 'DHR_Attn'
    #     # if file == 'train_points_rigid_img':
    #     #     runs.append(['python', 'train_points_rigid.py', '--dataset', str(data), 
    #     #         '--model', str(model), '--num_epochs', str(10), 
    #     #         '--learning_rate', str(lr), '--decay_rate', str(decay_rate),
    #     #         '--image', str(1), '--points', str(0), '--sup', str(sup)])
    #     # elif file == 'train_points_rigid_pt':
    #     #     runs.append(['python', 'train_points_rigid.py', '--dataset', str(data), 
    #     #         '--model', str(model), '--num_epochs', str(10), 
    #     #         '--learning_rate', str(lr), '--decay_rate', str(decay_rate),
    #     #         '--image', str(0), '--points', str(1), '--sup', str(sup)])
    #     if file == 'train_img_batch.py':
    #         runs.append(['python', 'train_img_batch.py', '--dataset', str(data), 
    #             '--model', str(model), '--num_epochs', str(5), 
    #             '--learning_rate', str(lr), '--decay_rate', str(decay_rate),
    #             '--image', str(1), '--points', str(0), '--sup', str(sup)])
        # elif file == 'train_one_sample.py':
        #     runs.append(['python', 'train_one_sample.py', '--dataset', str(data), 
        #         '--model', str(model), '--num_epochs', str(10), 
        #         '--learning_rate', str(lr), '--decay_rate', str(decay_rate),
        #         '--image', str(1), '--points', str(0), '--loss_image', str(0),
        #         '--sup', str(sup)])
        # runs.append(['python', str(file), '--dataset', str(data), 
        #                 '--model', str(model), 
        #                 '--image', str(1), '--points', str(0), '--sup', str(sup)])
        # runs.append(['python', str(file), '--dataset', str(data), 
        #                 '--model', 'DHR', '--num_epochs', str(100), 
        #                 '--image', str(1), '--points', str(0), '--sup', str(sup)])
        # runs.append(['python', str(file), '--dataset', str(data), 
        #     '--model', str(model), '--num_epochs', str(20), 
        #     '--image', str(1), '--points', str(0), '--loss_image', str(0),
        #     '--sup', str(sup)])

# for model in models:
#     runs.append(['python', 'train_stage.py', '--learning_rate', str(1e-3), '--decay_rate', str(0.5e-3),
#                     '--model', str(model), '--num_epochs', str(20), 
#                     '--image', str(1), '--points', str(0), '--loss_image', str(0),
#                     ])
        
dup_runs = 1
for i in range(len(runs)):
    for j in range(dup_runs):
        print(f"\n{runs[i]}")
        subprocess.run(runs[i])
    
print("Total runs: ", len(runs)*dup_runs)

''' notes
train_points_rigid.py - no batch training
train_img_batch.py - batch training
train_one_sample.py - one sample training
'''