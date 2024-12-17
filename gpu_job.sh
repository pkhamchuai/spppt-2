#!/bin/bash
#SBATCH --job-name=spppt        ## ชื่อของงาน
#SBATCH --output=hpc_output_error/output_%j.out    ## ชื่อไฟล์ Output (%j = Job-ID)
#SBATCH --error=hpc_output_error/error_%j.err     ## ชื่อไฟล์ error (%j = Job-ID)
#SBATCH --partition=mixed
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus=1                 # total number of GPUs
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --ntasks-per-node=1      # number of tasks per node

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=1   #$SLURM_NNODES * $SLURM_NTASKS_PER_NODE
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# source /home/pakpoom_kh/.bashrc
source .venv/bin/activate
# python test_ensemble.py
python test_BCS_erawan.py
# python test_ensemble_DHR2x.py
# python test.py
# python test1.py
# python test2.py
# python run_SP_batch.py 