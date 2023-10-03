FIRST=0 #${1}          # 0
LAST=36114 #${2}           # 36114
STEP=6100 #${3}           # 6100

PROJECTIONS_PATH="/mnt/proj1/open-26-3/datasets/nuscenes/features/projections"
SAVE_DIR="/scratch/project/dd-23-68/vobecant/datasets/nuscenes/ovseg_features"

for START in $(seq ${FIRST} ${STEP} ${LAST}); do
  END=$((${START} + ${STEP}))

  EXPNAME=val_inference_${START}_${END}
  JOB_FILE=./jobs/${EXPNAME}.job

  echo "#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name ${EXPNAME}
#SBATCH --account DD-23-68
#SBATCH --output=${EXPNAME}.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --partition=qgpu
#SBATCH --time=10:00:00
#SBATCH --signal=SIGUSR1@90

cd /scratch/project/dd-23-68/vobecant/projects/ovseg
module load CUDA/11.7
module load GCC/11.3.0


module load CUDA/11.7
module load GCC/11.3.0

. /scratch/project/dd-23-68/miniconda3/etc/profile.d/conda.sh
conda activate ovseg

export CUDA_VISIBLE_DEVICES='0'

# run script from above
srun python -u inference.py --save-dir ${SAVE_DIR} --projections-path ${PROJECTIONS_PATH} --debug --points-given --verbose --no-multimask --start-end ${START} ${END}" >${JOB_FILE}
  echo "run job ${JOB_FILE}"
  sbatch ${JOB_FILE}
  echo ""
  sleep 1
done

squeue -u vobecant
