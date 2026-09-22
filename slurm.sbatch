#!/bin/bash -l
#SBATCH -J MHKP-V2-T13
# Table 13: submodel V2 on the multi-bin grid of Deplano et al. (2019),
#   items {70,100,130,160,200} x bins {2,3,5} x 5 instances x 4 solvers
#     = 300 array tasks.
# The fourth solver is wfbf-grid, the literal reading of Algorithm 1 line
# 15; wfbf-corner is the restricted one the earlier 225-task array ran.
# Must match main_slurm.py's grid: python3 main_slurm.py --print-grid
#SBATCH --array=0-74
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
# MEMORY. Sized for the worst task in the array, which is the MILP: V2 is
# 10n^2 + 4nm + m + 12n variables and its constraint count carries an
# n^2 m^2 term, so at n=200, m=5 the model is large before Gurobi even
# starts branching. WFBF and the beam search need a small fraction of this,
# but a heterogeneous array has to request the maximum.
#SBATCH --mem=32G
#
# WALL TIME. Per array task, not for the array as a whole. Two tasks now
# reach 3600 s: the MILP at its solve limit - model build and Gurobi's own
# presolve sit OUTSIDE that limit and have been seen to take considerably
# longer than expected on these sizes - and wfbf-grid, which is capped at
# 3600 s from inside main_slurm.py. Hence 3600 s plus a wide margin rather
# than 3600 s exactly. For scale on the grid task: literal enumeration at
# beta = 1 costs ~87 s for ONE bin of 70 items (measured, 120^3 bin), and
# these configurations run 2-5 bins of up to 200 items.
#SBATCH --time=02:30:00
#
#SBATCH -o logs/%x-%A-%a.out
#SBATCH -e logs/%x-%A-%a.err
#SBATCH --mail-type=END,FAIL
##SBATCH --constraint="XEON_E5_2637v3"
##SBATCH --constraint="EPYC_9135"

##module load python/3.13
##module load gcc/latest
##module load gurobipy/12.0.3
source /home/mcaspar/3DPCPTP/bin/activate
export GRB_LICENSE_FILE=/software/gurobi/gurobi.lic

# Gurobi and BLAS both honour these; the beam search is pure Python. One
# thread per task, because the array already uses the node's cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

# The per-task reports are written by main_slurm.py itself, one json each, so
# nothing has to be collected while the array runs. The summary job below
# waits for the whole array and prints Table 13 from those reports.
#
# afterany: it also runs when some tasks failed, and then reports which
# configurations are incomplete instead of silently averaging over a partial
# grid. Submitted from task 0 before the solve, so it is queued even if this
# task is later preempted.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm.py --summarize \
--csv ${SLURM_SUBMIT_DIR}/results_table13_V2.csv"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-V2-T13-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-V2-T13-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: \
python3 main_slurm.py --summarize"
fi

# Solver and time limit come from the task id: main_slurm.py maps it to a
# (configuration, instance, solver) triple and applies that solver's own
# budget - uncapped for wfbf-corner, 3600 s for wfbf-grid and the MILP,
# 900 s for the beam search. Pass --time-limit here only to override all
# four at once, for a sweep.
python3 main_slurm.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table13_V2"
