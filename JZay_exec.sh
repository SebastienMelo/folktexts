#!/bin/bash

#SBATCH --job-name=folktexts-gpu
#SBATCH -o outslurm/gpu/job%A_%a.out
#SBATCH -e outslurm/gpu/job%A_%a.err


#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

#SBATCH -A nuj@h100
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

#SBATCH --array=0-1

echo "------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"  
echo "Run on host: "`hostname` 
echo "Operating system: "`uname -s` 
echo "Username: "`whoami` 
echo "Started at: "`date` 
echo "------------------------------------------------" 



module purge
module load arch/h100
module load miniforge/24.9.0

conda activate folktexts


A_VALUES=('google/gemma-3-4b-it' 'meta-llama/Llama-3.2-1B')



srun run_acs_benchmark --model ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --results-dir folktexts-results --data-dir data