#!/bin/bash -l
#SBATCH -J MHKP-T15-BS
# Table 15-inspired analog: V2 beam search only, 4 item counts x 10 repeats
# = 40 array tasks. Each task has a 300-second search budget.
# Must match main_slurm15.py: python3 main_slurm15.py --print-grid
#SBATCH --array=0-39
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# Budget plus startup, input generation, and report-writing overhead.
#SBATCH --time=00:10:00
#SBATCH -o logs/%x-%A-%a.out
#SBATCH -e logs/%x-%A-%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="XEON_E5_2637v3"
##SBATCH --constraint="EPYC_9135"

source /home/mcaspar/3DPCPTP/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

# The summary runs after the full array, even if some tasks fail, so missing
# or partial results remain visible. It reads the per-task JSON reports.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; \
python3 ${SLURM_SUBMIT_DIR}/main_slurm15.py --summarize \
--results-dir ${SLURM_SUBMIT_DIR}/results_table15_bs"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=MHKP-T15-BS-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
           --output=logs/MHKP-T15-BS-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: \
python3 main_slurm15.py --summarize"
fi

python3 "${SLURM_SUBMIT_DIR}/main_slurm15.py" \
    --id "${SLURM_ARRAY_TASK_ID}" \
    --time-limit 300 \
    --results-dir "${SLURM_SUBMIT_DIR}/results_table15_bs"
