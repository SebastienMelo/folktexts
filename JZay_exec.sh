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
#SBATCH --time=70:00:00
#SBATCH --qos=qos_gpu_h100-t4

#SBATCH --array=0-2

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

A_VALUES=('meta-llama/Meta-Llama-3-70B' 'meta-llama/Llama-2-70b-hf' 'Qwen/Qwen2.5-72B-Instruct')


python -m folktexts.cli.run_acs_benchmark --model $model_dir${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --data-dir data --results-dir folktexts-results --batch-size 32
# srun run_acs_benchmark --model ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --results-dir folktexts-results --data-dir data