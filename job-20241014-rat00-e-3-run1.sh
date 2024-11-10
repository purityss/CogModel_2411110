#!/bin/bash
#SBATCH -A hpc1406182255
#SBATCH --partition=GPU40G
#SBATCH --qos=normal
#SBATCH -J job-20241014-rat00-e-3-run1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=180
#SBATCH --chdir=/lustre/home/2401111678/CogModelingRNNsTutorial-main
#SBATCH --output=rat00-e-3-run1-job.%j.out
#SBATCH --error=rat00-e-3-run1-job.%j.err
#SBATCH --gres=gpu:1

source /etc/profile.d/modules.sh
module load anaconda3/2023.03.01
module load cuda
source activate python39
python main_RAT_remote_e-3.py