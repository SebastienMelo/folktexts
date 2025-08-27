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

#SBATCH --array=0-18

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

A_VALUES=('01-ai/Yi-1.5-34B-Chat' '01-ai/Yi-1.5-34B' '01-ai/Yi-1.5-9B-Chat' '01-ai/Yi-1.5-9B' 
'google/gemma-2-9b' 'google/gemma-2-27b' 'google/gemma-2-27b-it'
'mistralai/Mistral-Small-3.2-24B-Instruct-2506' 'mistralai/Mistral-Small-3.1-24B-Base-2503' 'mistralai/Mistral-7B-Instruct-v0.3' 'mistralai/Mistral-7B-v0.3'
'meta-llama/Llama-2-13b-chat-hf' 'meta-llama/Llama-3.2-1B-Instruct' 'meta-llama/Llama-3.2-3B' 'meta-llama/Llama-3.2-3B-Instruct' 'meta-llama/Llama-Guard-3-1B' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-Guard-3-8B'
'microsoft/Orca-2-7b')


python -m folktexts.cli.run_acs_benchmark --model $model_dir${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --data-dir data --results-dir folktexts-results --batch-size 32
# srun run_acs_benchmark --model ${A_VALUES[$SLURM_ARRAY_TASK_ID]} --task ACSIncome --results-dir folktexts-results --data-dir data