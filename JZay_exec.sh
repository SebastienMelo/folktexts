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
#SBATCH --time=90:00:00
#SBATCH --qos=qos_gpu_h100-t4

#SBATCH --array=0-26

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

A_VALUES=('google/gemma-2-9b' 'google/gemma-2-27b' 'google/gemma-2-27b-it' 'google/gemma-2-9b-it'
'mistralai/Mistral-7B-Instruct-v0.3' 'mistralai/Mistral-7B-v0.3' 'mistralai/Mixtral-8x7B-Instruct-v0.1' 'mistralai/Mixtral-8x7B-v0.1'
'meta-llama/Llama-2-13b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Llama-2-70b-chat-hf' 'meta-llama/Llama-2-70b-hf'
'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-3.1-8B-Instruct' 'meta-llama/Llama-3.2-1B' 'meta-llama/Llama-3.2-1B-Instruct'
'meta-llama/Llama-3.2-3B' 'meta-llama/Llama-3.2-3B-Instruct' 'meta-llama/Llama-3.3-70B-Instruct' 'meta-llama/Meta-Llama-3-70B'
'meta-llama/Meta-Llama-3-70B-Instruct' 'meta-llama/Meta-Llama-3-8B' 'meta-llama/Meta-Llama-3-8B-Instruct'
'microsoft/Orca-2-13b' 'microsoft/phi-4'
'deepseekai/DeepSeek-R1-Distill-Llama-8B' 'deepseekai/DeepSeek-R1-Distill-Llama-70B')

python -m folktexts.cli.run_acs_benchmark --model $model_dir${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task 'ACSEmployment' --data-dir data --results-dir folktexts-results-ICML --batch-size 16 --subsampling 0.5 --seed 42
# srun run_acs_benchmark --model ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --results-dir folktexts-results --data-dir data