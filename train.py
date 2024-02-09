import subprocess

dataset = [3]
models = ['DHR']
# models = ['SP_AffineNet4', 'DHR']
sups = [1, 0]
runs = []
# learning_rate = 1e-4
# loss = [[1, 1]]
# offset = 0.01
# sigma = range(30, 100, 10)

# generate run commands
for sup in sups:
    for data in dataset:
        if data == 0 and sup == 1:
            pass
        else:
            # runs.append(['python', 'train_points_rigid.py', '--dataset', str(data), 
            #              '--model', 'DHR', 
            #              '--image', str(1), '--points', str(0), '--sup', str(sup)])
            # runs.append(['python', 'train_one_sample.py', '--dataset', str(data), 
            #              '--model', 'DHR', '--num_epochs', str(100), 
            #              '--image', str(1), '--points', str(0), '--sup', str(sup)])
            runs.append(['python', 'train_img_batch.py', '--dataset', str(data), 
                            '--model', 'DHR', '--num_epochs', str(10), 
                            '--image', str(1), '--points', str(0), '--sup', str(sup)])
    

for i in range(len(runs)):
    for j in range(1):
        print(runs[i])
        subprocess.run(runs[i])