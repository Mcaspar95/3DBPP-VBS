#!/bin/bash -l
#SBATCH -J MHKP-V2-T12
# Table 12 follow-up: n = 100-200 (t12_n* groups), WFBF-GRID AND BS ONLY.
#
# WHY MILP IS SKIPPED HERE: the V2 MILP has a confirmed defect - for
# n >= 27 on the Table 11 sweep (n = 18-90, see
# table_results_table11_v2.tex) it returns the EMPTY packing as status
# OPTIMAL with a zero gap in a few seconds, which beam search's verified
# feasible packings on the same instances contradict. n = 100 already
# reproduces it (checked directly before this array was written), so this
# run solves only wfbf-grid and bs. Once the MILP defect is root-caused
# and fixed, the 110 missing MILP tasks (ids 690-1019 where n_items in
# {100,...,200} and solver=milp) can be run separately with
# slurm_t11t12.sbatch and --solver milp appended.
#
# The full grid still has ids 690-1019 for n = 100-200 (see
# main_slurm_t11t12.py --print-grid); every third id in that range maps to
# a milp entry. --solver, PASSED TOGETHER WITH --id, makes
# main_slurm_t11t12.py SKIP a task whose own grid entry is not that
# solver, printing one line and writing no report, rather than solving it
# anyway - see the "--id ... --solver" handling added to main.py for this
# purpose. So each array task below tries wfbf-grid, and if that was not
# its assigned solver, tries bs instead; a milp-assigned task skips both
# and does nothing, cheaply.
#
# Must match main_slurm_t11t12.py's grid: python3 main_slurm_t11t12.py --print-grid
#SBATCH --array=690-1019
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
# MEMORY. Neither wfbf-grid nor bs needs anywhere near the MILP's
# footprint (no Gurobi model is ever built in this run), but kept at
# 16G rather than trimmed further, since both are pure Python and the
# difference in scheduling cost is negligible on this cluster.
#SBATCH --mem=16G
#
# WALL TIME. wfbf-grid is capped at 900 s from inside
# main_slurm_t11t12.py via SIGALRM; bs is capped at 900 s too. A
# milp-assigned task that skips both costs seconds. 35 minutes covers
# the slower of the two real solves plus margin; neither needs the
# MILP's 3600 s allowance.
#SBATCH --time=00:35:00
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

# Summary job, queued from the first array task before the solve so it is
# submitted even if that task is later preempted. afterany also runs when
# some tasks failed, so incomplete sizes are reported rather than silently
# averaged over.
if [ "${SLURM_ARRAY_TASK_ID}" = "690" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm_t11t12.py --summarize \
--csv ${SLURM_SUBMIT_DIR}/results_table11_12_V2.csv"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-V2-T12-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-V2-T12-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: \
python3 main_slurm_t11t12.py --summarize"
fi

# --id maps this task to its (n_items, instance, solver) triple, cycling
# milp / wfbf-grid / bs across consecutive ids (see build_grid's
# ordering). --solver, given alongside --id, makes the task SKIP if its
# own entry is not that solver, rather than run it anyway - see the
# comment above. Each task therefore makes two attempts, one per solver
# wanted here; whichever matches the task's own entry runs, the other
# call is a one-line skip.
python3 main_slurm_t11t12.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --solver wfbf-grid \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table11_12_V2"

python3 main_slurm_t11t12.py \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --solver bs \
    --num_cpu "${SLURM_CPUS_PER_TASK}" \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table11_12_V2"
