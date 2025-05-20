#!/bin/bash
#PBS -N diffusion
#PBS -o job_out.log
#PBS -e job_err.log
#PBS -l nodes=gpu-h100:ppn=40

cd $PBS_O_WORKDIR

source ../venv/bin/activate
python twist_shift.py &
python twist_shift.py --bounded true &
python twist_weighted.py &
python twist_weighted.py --bounded true &

wait
