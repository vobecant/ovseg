FIRST=0
LAST=168780
STEP=21100

for START in $(seq ${FIRST} ${STEP} ${LAST}); do
  END=$((${START} + ${STEP}))
  EXPNAME=${START}_${END}_odise_features
  JOB_FILE=./jobs/${EXPNAME}.job
  echo "#!/bin/bash
#SBATCH --job-name ${EXPNAME}
#SBATCH --account DD-23-68
#SBATCH --output=${EXPNAME}.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --partition=qgpu
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@90

echo $(pwd)

cd /scratch/project/dd-23-68/vobecant/projects/ODISE

module load CUDA/11.7
module load GCC/11.3.0

. /scratch/project/dd-23-68/miniconda3/etc/profile.d/conda.sh
conda activate odise

export CUDA_VISIBLE_DEVICES='0'

python3 -u demo/nuscenes_inference.py --config-file configs/Panoptic/odise_label_coco_50e.py --image-list-path /scratch/project/dd-23-68/vobecant/projects/TPVFormer-OpenSet/data/train_paths.txt --save-dir /scratch/project/dd-23-68/vobecant/projects/PhD/ODISE/out/nuscenes_features --fts-only --start-idx ${START} --end-idx ${END}" >${JOB_FILE}
  echo "run ${JOB_FILE}"
  sbatch ${JOB_FILE}
done

echo ""
sleep 1
squeue -u vobecant
