#!/bin/bash

#SBATCH --job-name=folktexts-gpu
#SBATCH -o outslurm/gpu/job%A_%a.out
#SBATCH -e outslurm/gpu/job%A_%a.err



#SBATCH --cpus-per-task=24
#SBATCH -A nuj@cpu

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
module load pytorch-gpu/py3/2.7.0


python -m preprocess_tables_metamodel