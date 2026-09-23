#!/bin/bash -l
#SBATCH -J MHKP-V2-T1112
# Tables 11+12: submodel V2 on the single-bin size sweep of Deplano et al.
# (2019), n = 18-90 (Table 11, t8_n* groups) and n = 100-200 step 10
# (Table 12, t12_n* groups), 34 sizes x 10 instances x 3 solvers
#   = 1020 array tasks.
# Must match main_slurm_t11t12.py's grid: python3 main_slurm_t11t12.py --print-grid
#SBATCH --array=0-1019
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
# MEMORY. Sized for the worst task in the array, which is the MILP at
# n=200: V2 is 10n^2 + 4nm + m + 12n variables with an n^2m^2 constraint
# term (m=1 here, so that term is tame, but n^2 alone is already ~40000
# at n=200). WFBF-grid and the beam search need a small fraction of this,
# but a heterogeneous array has to request the maximum.
#SBATCH --mem=16G
#
# WALL TIME. Per array task, not for the array as a whole. The longest
# task is the MILP at its 3600 s solve limit; model build and Gurobi's
# own presolve sit OUTSIDE that limit and have been seen to take longer
# than expected on these sizes, hence 3600 s plus a wide margin rather
# than 3600 s exactly. wfbf-grid is capped at 900 s from inside
# main_slurm_t11t12.py via SIGALRM.
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

# The per-task reports are written by main_slurm_t11t12.py itself, one json
# each, so nothing has to be collected while the array runs. The summary job
# below waits for the whole array and prints the Table 11/12 layout from
# those reports.
#
# afterany: it also runs when some tasks failed, and then reports which
# sizes are incomplete instead of silently averaging over a partial grid.
# Submitted from task 0 before the solve, so it is queued even if this task
# is later preempted.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm_t11t12.py --summarize \
--csv ${SLURM_SUBMIT_DIR}/results_table11_12_V2.csv"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-V2-T1112-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-V2-T1112-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: \
python3 main_slurm_t11t12.py --summarize"
fi

# Solver and time limit come from the task id: main_slurm_t11t12.py maps it
# to an (n_items, instance, solver) triple and applies that solver's own
# budget - 3600 s for the MILP, 900 s for wfbf-grid and the beam search.
# Pass --time-limit here only to override all three at once, for a sweep.
python3 main_slurm_t11t12.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table11_12_V2"
