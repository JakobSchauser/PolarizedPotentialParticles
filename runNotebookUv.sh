#!/bin/bash -l
PARTITION="main"
GPUS="1"
CPUS="16"
MEM="64G"
TIME="04:00:00"
PORT="8888"


# Activate the virtual environment
source $HOME/PolarizedPotentialParticles/.venv/bin/activate

# Launch an interactive Jupyter Lab session inside an srun allocation
srun --partition="$PARTITION" \
	--gres=gpu:"$GPUS" \
	--cpus-per-task="$CPUS" \
	--mem="$MEM" \
	--time="$TIME" \
	--pty \
	jupyter lab --no-browser --ip=0.0.0.0 --port="$PORT"