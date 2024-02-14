#!/bin/bash
#SBATCH --job-name=spppt        ## ชื่อของงาน
#SBATCH --output=output_%j.out    ## ชื่อไฟล์ Output (%j = Job-ID)
#SBATCH --error=error_%j.err     ## ชื่อไฟล์ error (%j = Job-ID)
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus=1                 # total number of GPUs
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1              ## จำนวน tasks ที่ต้องการใช้ในการรัน

#Test models on dataset 3 (new)
module purge
module load anaconda3 cuda/11.8
python test.py