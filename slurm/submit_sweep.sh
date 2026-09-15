#!/usr/bin/env bash
# =============================================================================
# Submits the sweep on HPC@PoliTO Legion, packing 4 runs per A40 node.
#
#   1. staging job (cpu_sapphire): copies code + dataset to $SCRATCH_FLASH and
#      validates all 20 category/seed combinations with --check-only
#   2. PatchCore array   (5 tasks x 4 GPUs = 20 runs), --dependency=afterok
#   3. EfficientAD array (5 tasks x 4 GPUs = 20 runs), same dependency
#
# The grid check runs inside staging on purpose: with afterok, a broken
# category aborts the whole sweep before a single full node is allocated.
#
#   ./slurm/submit_sweep.sh
#   DATASET_SRC=$HOME/datasets/mvtec ./slurm/submit_sweep.sh
#   MAX_PARALLEL=3 ./slurm/submit_sweep.sh          # 3 nodes at a time per model
#   STAGE=0 SWEEP_ID=ablation_dust ./slurm/submit_sweep.sh
#   EAD_PARTITION=gpu_a40_ext EAD_TIME=2-00:00:00 ./slurm/submit_sweep.sh
# =============================================================================

set -euo pipefail

: "${SCRATCH_FLASH:?SCRATCH_FLASH is not set - log in to Legion first}"

# --- ADATTA AI TUOI PERCORSI -------------------------------------------------
export HOME_REPO="${HOME_REPO:-$PWD}"
export DATASET_SRC="${DATASET_SRC:-$HOME/TESI_EMA/anomaly_detection_for_textile_industry/data}"
export IMAGENETTE_SRC="${IMAGENETTE_SRC:-$HOME/TESI_EMA/anomaly_detection_for_textile_industry/data/imagenette_for_efficientad}"
MAIL="${MAIL:-s346291@studenti.polito.it}"
# -----------------------------------------------------------------------------

export SWEEP_ID="${SWEEP_ID:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SCRATCH="${SCRATCH_FLASH}/ad_sweeps/${SWEEP_ID}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
STAGE="${STAGE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

# The grid must match the one in train_sweep.sbatch.
CATEGORIES=(carpet no_dust dustOnValidation dustOnValidationAndTrain)
SEEDS=(0 1 2 42 101)
N_RUNS=$(( ${#CATEGORIES[@]} * ${#SEEDS[@]} ))
# Ceiling division: a partially filled last node still needs a task.
N_TASKS=$(( (N_RUNS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))
ARRAY_SPEC="0-$(( N_TASKS - 1 ))%${MAX_PARALLEL}"

# Each task runs 4 models in parallel, so walltime is the slowest single run
# plus contention, not 4x. gpu_a40 caps at 24 h.
PC_PARTITION="${PC_PARTITION:-gpu_a40}"
PC_TIME="${PC_TIME:-0-06:00:00}"
EAD_PARTITION="${EAD_PARTITION:-gpu_a40}"
EAD_TIME="${EAD_TIME:-0-23:00:00}"

if [[ ! -d "${DATASET_SRC}" ]]; then
    echo "[error] dataset not found: ${DATASET_SRC}" >&2
    echo "        set DATASET_SRC to the directory containing the 4 categories." >&2
    exit 2
fi

mkdir -p slurm_logs "sweeps/${SWEEP_ID}"

echo "sweep id      : ${SWEEP_ID}"
echo "repo (\$HOME)  : ${HOME_REPO}"
echo "scratch       : ${SWEEP_SCRATCH}"
echo "dataset src   : ${DATASET_SRC}"
echo "grid          : ${#CATEGORIES[@]} categories x ${#SEEDS[@]} seeds = ${N_RUNS} runs per model"
echo "packing       : ${GPUS_PER_JOB} runs/node, array ${ARRAY_SPEC}"
echo

DEP=""
if [[ "${STAGE}" == "1" ]]; then
    JID_STAGE=$(sbatch --parsable \
        --job-name="ad_stage_${SWEEP_ID}" \
        --partition=cpu_sapphire \
        --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=4 \
        --mem=8G --time=0-01:00:00 \
        --output="slurm_logs/%x_%j.out" --error="slurm_logs/%x_%j.err" \
        --mail-type=FAIL --mail-user="${MAIL}" \
        --export=ALL,HOME_REPO,DATASET_SRC,IMAGENETTE_SRC,SWEEP_SCRATCH \
        <<'STAGE_EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "[stage] target: ${SWEEP_SCRATCH}"
mkdir -p "${SWEEP_SCRATCH}/code" "${SWEEP_SCRATCH}/data" "${SWEEP_SCRATCH}/configs"

rsync -a --delete \
      --exclude '.git' --exclude 'sweeps' --exclude 'slurm_logs' \
      --exclude 'data' --exclude '__pycache__' --exclude 'results' \
      "${HOME_REPO}/" "${SWEEP_SCRATCH}/code/"

rsync -a "${DATASET_SRC}/" "${SWEEP_SCRATCH}/data/mvtec/"

if [[ -d "${IMAGENETTE_SRC}" ]]; then
    rsync -a "${IMAGENETTE_SRC}/" "${SWEEP_SCRATCH}/data/imagenette_for_efficientad/"
else
    echo "[stage] WARNING: ${IMAGENETTE_SRC} not found - EfficientAD will fail pre-flight"
fi

echo "[stage] categories staged:"
ls -1 "${SWEEP_SCRATCH}/data/mvtec"
du -sh "${SWEEP_SCRATCH}/data"

# ---- validate the whole grid before any GPU node is allocated ----
echo "[stage] validating the 20-cell grid ..."
cd "${SWEEP_SCRATCH}/code"
BAD=0
for MODEL in patchcore efficientad; do
    for CATEGORY in carpet no_dust dustOnValidation dustOnValidationAndTrain; do
        python slurm/make_run_config.py --check-only \
            --model "${MODEL}" --category "${CATEGORY}" --seed 0 \
            --dataset-root "${SWEEP_SCRATCH}/data/mvtec" \
            --imagenette-dir "${SWEEP_SCRATCH}/data/imagenette_for_efficientad" \
            || BAD=1
    done
done
if [[ ${BAD} -ne 0 ]]; then
    echo "[stage] grid validation FAILED - dependent training jobs will be cancelled" >&2
    exit 1
fi
echo "[stage] done"
STAGE_EOF
)
    echo "staging       : job ${JID_STAGE} (cpu_sapphire, includes grid check)"
    DEP="--dependency=afterok:${JID_STAGE}"
else
    echo "staging       : skipped (STAGE=0)"
fi

JID_PC=$(sbatch --parsable ${DEP} \
    --job-name="ad_patchcore_${SWEEP_ID}" \
    --array="${ARRAY_SPEC}" \
    --partition="${PC_PARTITION}" \
    --time="${PC_TIME}" \
    --mail-user="${MAIL}" \
    --export=ALL,SWEEP_ID,HOME_REPO,GPUS_PER_JOB \
    slurm/train_sweep.sbatch patchcore)
echo "patchcore     : array ${JID_PC}  ${N_TASKS} nodes x ${GPUS_PER_JOB} runs  (${PC_PARTITION}, ${PC_TIME})"

JID_EAD=$(sbatch --parsable ${DEP} \
    --job-name="ad_efficientad_${SWEEP_ID}" \
    --array="${ARRAY_SPEC}" \
    --partition="${EAD_PARTITION}" \
    --time="${EAD_TIME}" \
    --mail-user="${MAIL}" \
    --export=ALL,SWEEP_ID,HOME_REPO,GPUS_PER_JOB \
    slurm/train_sweep.sbatch efficientad)
echo "efficientad   : array ${JID_EAD}  ${N_TASKS} nodes x ${GPUS_PER_JOB} runs  (${EAD_PARTITION}, ${EAD_TIME})"

cat <<INFO

monitor    : squeue -u \$USER
per run    : grep RESULT slurm_logs/ad_*_${SWEEP_ID}_*.out
details    : sacct -j ${JID_PC} --format=JobID,State,Elapsed,MaxRSS,ExitCode
cancel all : scancel ${JID_PC} ${JID_EAD}

results land in ${HOME_REPO}/sweeps/${SWEEP_ID}/<model>/<category>/seed<N>/
aggregate with: python slurm/aggregate_results.py sweeps/${SWEEP_ID}
INFO
