FIRST=0 #${1}          # 0
LAST=36114 #${2}           # 36114
STEP=6100 #${3}           # 6100

PROJECTIONS_PATH="/nfs/datasets/nuscenes/features/projections"
SAVE_DIR="/nfs/datasets/nuscenes/ovseg_nuscenes_val_predictions"

for START in $(seq ${FIRST} ${STEP} ${LAST}); do
  END=$((${START} + ${STEP}))

  EXPNAME=val_inference_${START}_${END}
  JOB_FILE=./jobs/${EXPNAME}.job

  echo "#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --output=${EXPNAME}.err
#SBATCH --mem=40GB
#SBATCH --time=10:00:00
#SBATCH --exclude=node-10,node-12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --signal=SIGUSR1@90

# activate conda env
module purge
source activate /home/vobecant/miniconda3/envs/ovseg

# run script from above
srun python -u inference.py --save-dir ${SAVE_DIR} --projections-path ${PROJECTIONS_PATH} --debug --points-given --verbose --no-multimask --start-end ${START} ${END}" >${JOB_FILE}
  echo "run job ${JOB_FILE}"
  sbatch ${JOB_FILE}
  echo ""
  sleep 1
done

squeue -u vobecant
