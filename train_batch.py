import subprocess

dataset = [1, 2, 0]
sups = [1, 1, 0]
# models = ['DHR', 'SP_AffineNet']
# models = ['SP_AffineNet4', 'DHR']
runs = []
learning_rate = 1e-4
loss = [[1, 1]]
offset = 0.01
sigma = range(30, 100, 10)
# generate run commands
for j in sigma:
    

for i in range(len(runs)):
    print(runs[i])
    # subprocess.run(runs[i])