#!/bin/bash -l
#SBATCH -J MHKP-T10-V1
# Table 10 V1: items {70,100,130,160,200} x bins {2,3,5}
# x 5 instances x {MILP, BS} = 150 tasks.
# Keep synchronized with: python3 main_slurm_table10_v1.py --print-grid
#SBATCH --array=0-149
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
# MILP solve limit is 3600s; extra time covers model construction/cleanup.
#SBATCH --time=02:30:00
#SBATCH -o logs/%x-%A-%a.out
#SBATCH -e logs/%x-%A-%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="XEON_E5_2637v3"
##SBATCH --constraint="EPYC_9135"

##module load python/3.13
##module load gcc/latest
##module load gurobipy/12.0.3
source /home/mcaspar/3DPCPTP/bin/activate
export GRB_LICENSE_FILE=/software/gurobi/gurobi.lic
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

# Queue one summary job after the complete array, including if some tasks fail.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm_table10_v1.py --summarize \
--results-dir ${SLURM_SUBMIT_DIR}/results_table10_v1 \
--csv ${SLURM_SUBMIT_DIR}/results_table10_v1.csv"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-T10-V1-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-T10-V1-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; after the array run: \
python3 main_slurm_table10_v1.py --summarize"
fi

python3 main_slurm_table10_v1.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table10_v1"
