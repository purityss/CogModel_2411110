#!/bin/bash
#SBATCH -A hpc1406182255
#SBATCH --partition=GPU80G
#SBATCH --qos=normal
#SBATCH -J job-20241014-LACA-e-3-run1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=120
#SBATCH --chdir=/lustre/home/2401111678/CogModelingRNNsTutorial-main
#SBATCH --output=LACA-e-3-run1-job.%j.out
#SBATCH --error=LACA-e-3-run1-job.%j.err
#SBATCH --gres=gpu:1

source /etc/profile.d/modules.sh
module load anaconda3/2023.03.01
module load cuda
source activate python39
python main_LACA_remote_e-3.py