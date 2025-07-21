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

#SBATCH --array=0-7

echo "------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"  
echo "Run on host: "`hostname` 
echo "Operating system: "`uname -s` 
echo "Username: "`whoami` 
echo "Started at: "`date` 
echo "------------------------------------------------" 



module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0


model_dir=$DSDIR/HuggingFace_Models/

A_VALUES=('meta-llama/Llama-3-70B-Instruct' 'Qwen/Qwen2.5-VL-7B-Instruct'
         'Qwen/Qwen2.5-VL-32B-Instruct' 'Qwen/Qwen2.5-72B-Instruct'
         'google/gemma-2b-it' 'google/gemma-3-4b-it'
         'google/gemma-2-9b-it' 'google/gemma-3-27b-it')


python -m folktexts.cli.run_acs_benchmark --model $model_dir${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --data-dir data --results-dir folktexts-results
# srun run_acs_benchmark --model ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --results-dir folktexts-results --data-dir data