#!/bin/bash -l
#SBATCH -J MHKP-V2-T8
# V2 on the single-bin n=18..200 sweep: 34 sizes x 10 instances x 3 solvers
#   = 1020 array tasks. Must match main_slurm_t8.py --print-grid.
#SBATCH --array=0-1019
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
# Memory for the largest V2 MILP (n=200). WFBF and BS use much less, but
# this is a heterogeneous array.
#SBATCH --mem=16G
#
# Per-task wall time: the MILP solve budget is 3600 s, plus extra margin for
# model construction and solver startup. BS has a 60 s budget; corner-mode
# WFBF is a deterministic uncapped constructive pass.
#SBATCH --time=01:30:00
#
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

# Submit a dependent summary job once; it runs even if array tasks fail and
# reports missing results rather than silently dropping incomplete sizes.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm_t8.py --summarize \
--csv ${SLURM_SUBMIT_DIR}/results_table8_V2.csv"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-V2-T8-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-V2-T8-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: \
python3 main_slurm_t8.py --summarize"
fi

python3 ${SLURM_SUBMIT_DIR}/main_slurm_t8.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table8_V2"
