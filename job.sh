#!/usr/bin/env bash 
#SBATCH -N 1                    
#SBATCH -n 1
#SBATCH -c 1 
#SBATCH --mem=24g                
#SBATCH -J "llm softprompt job"    
#SBATCH -p short                
#SBATCH -t 18:00:00             
#SBATCH --gres=gpu:1            
#SBATCH -C "A100"

srun --unbuffered python run_experiment.py "$@"