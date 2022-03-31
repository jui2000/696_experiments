#!/bin/bash
#SBATCH --job-name=696_512_256
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=2080ti-long             # Partition to submit to (serial_requeue)
#SBATCH --mem=4096               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=696_512_256/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=696_512_256/run_logs_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nidhichandra@cs.umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
set -x -e
##### Experiment settings #####
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/nfs/work1/mccallum/nmonath/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/nfs/work1/mccallum/nmonath/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/nfs/work1/mccallum/nmonath/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/nfs/work1/mccallum/nmonath/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate 696
sleep 1

wandb agent 696ds_deepmind/696_experiments/kkvpd63g

sleep 1
exit
