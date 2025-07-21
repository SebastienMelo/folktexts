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
module load pytorch-gpu/py3/2.7.0


model_dir=$DSDIR/HuggingFace_Models/

A_VALUES=('microsoft/phi-2' 'mistralai/Mixtral-8x22B-Instruct-v0.1' 'mistralai/Mixtral-8x7B-Instruct-v0.1' 
          'meta-llama/Llama-3.1-8B-Instruct' 'meta-llama/Llama-3.1-70B-Instruct' 
          'meta-llama/Llama-3.2-8B-Instruct' 'meta-llama/Llama-3.2-70B-Instruct')


python -m folktexts.cli.run_acs_benchmark --model $model_dir${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --data-dir data --results-dir folktexts-results
# srun run_acs_benchmark --model ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --results-dir folktexts-results --data-dir data