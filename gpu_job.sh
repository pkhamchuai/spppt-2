#!/bin/bash
#SBATCH --job-name=spppt        ## ชื่อของงาน
#SBATCH --output=hpc_output_error/output_%j.out    ## ชื่อไฟล์ Output (%j = Job-ID)
#SBATCH --error=hpc_output_error/error_%j.err     ## ชื่อไฟล์ error (%j = Job-ID)
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus=1                 # total number of GPUs
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1              ## จำนวน tasks ที่ต้องการใช้ในการรัน

# source /home/pakpoom_kh/.bashrc
source .venv/bin/activate
# python test_ensemble.py
python test_ensemble_DHRdiff.py
python test_ensemble_DHR2x.py
# python test.py
# python test1.py
# python test2.py
# python run_SP_batch.py 