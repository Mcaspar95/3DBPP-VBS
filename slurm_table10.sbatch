#!/bin/bash -l
#SBATCH -J MHKP-T10
# Table 10: our MILP and beam search under submodels V0, V1, V2 on the
# multi-bin grid of Deplano et al. (2019),
#   items {70,100,130,160,200} x bins {2,3,5} x 5 instances x 6 columns
#     (v0-milp, v0-bs, v1-milp, v1-bs, v2-milp, v2-bs)
#     = 450 array tasks.
# Must match main_slurm_table10.py's grid: python3 main_slurm_table10.py --print-grid
#SBATCH --array=0-449
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
# MEMORY. Sized for the worst task in the array, which is the V2 MILP at
# n=200, m=5: V2 is 10n^2 + 4nm + m + 12n variables and its constraint
# count carries an n^2m^2 term, the largest of the three submodels. V0's
# MILP and all three beam searches need a fraction of this, but a
# heterogeneous array has to request the maximum.
#SBATCH --mem=32G
#
# WALL TIME. Per array task, not for the array as a whole. The longest
# task is a MILP column at its 3600 s solve limit; model build and
# Gurobi's own presolve sit OUTSIDE that limit and have been seen to take
# considerably longer than expected on these sizes, hence 3600 s plus a
# wide margin rather than 3600 s exactly.
#SBATCH --time=02:30:00
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

# The per-task reports are written by main_slurm_table10.py itself, one
# json each, so nothing has to be collected while the array runs. The
# summary job below waits for the whole array and prints the Table 10
# layout (V0 | V1 | V2, each MILP + BS) from those reports.
#
# afterany: it also runs when some tasks failed, and then reports which
# configurations are incomplete instead of silently averaging over a
# partial grid. Submitted from task 0 before the solve, so it is queued
# even if this task is later preempted.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm_table10.py --summarize \
--csv ${SLURM_SUBMIT_DIR}/results_table10.csv"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-T10-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-T10-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: \
python3 main_slurm_table10.py --summarize"
fi

# Column and time limit come from the task id: main_slurm_table10.py maps
# it to an (n_items, n_bins, instance, column) quadruple and applies that
# column's own method budget - 3600 s for a *-milp column, 900 s for a
# *-bs column. Pass --time-limit here only to override all six at once,
# for a sweep.
python3 main_slurm_table10.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table10"
