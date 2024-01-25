import subprocess

dataset = [0, 1, 2, 3]
models = ['DHR']
# models = ['SP_AffineNet4', 'DHR']
sups = [1, 0]
runs = []
# learning_rate = 1e-4
# loss = [[1, 1]]
# offset = 0.01
# sigma = range(30, 100, 10)

# generate run commands
for j in sups:
    for i in dataset:
        if i == 0 and j == 1:
            pass
        else:
            runs.append(['python', 'train.py', '--dataset', str(dataset[i]), '--model', 'DHR', 
                         '--supervised', str(j)])
    

for i in range(len(runs)):
    for j in range(3):
        print(runs[i])
        # subprocess.run(runs[i])