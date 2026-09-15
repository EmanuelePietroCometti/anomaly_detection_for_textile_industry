#!/usr/bin/env bash
# =============================================================================
# Submits the sweep on HPC@PoliTO Legion, packing 4 runs per A40 node.
#
#   1. staging job (cpu_sapphire): copies code + data/ to $SCRATCH_FLASH and
#      validates every category with --check-only
#   2. PatchCore array   (5 tasks x 4 GPUs = 20 runs), --dependency=afterok
#   3. EfficientAD array (5 tasks x 4 GPUs = 20 runs), same dependency
#
# The grid check runs inside staging on purpose: with afterok, a broken
# category aborts the whole sweep before a single full node is allocated.
#
#   ./slurm/submit_sweep.sh
#   MAX_PARALLEL=3 ./slurm/submit_sweep.sh          # 3 nodes at a time per model
#   STAGE=0 SWEEP_ID=ablation_dust ./slurm/submit_sweep.sh
#   EAD_PARTITION=gpu_a40_ext EAD_TIME=2-00:00:00 ./slurm/submit_sweep.sh
#   SWEEP_CATEGORIES="carpet reda_baseline" ./slurm/submit_sweep.sh
# =============================================================================

set -euo pipefail

: "${SCRATCH_FLASH:?SCRATCH_FLASH is not set - log in to Legion first}"

# --- ADATTA AI TUOI PERCORSI -------------------------------------------------
export HOME_REPO="${HOME_REPO:-$PWD}"
# Categories and imagenette both live under <repo>/data, which the code rsync
# excludes and this one copies wholesale.
export DATA_SRC="${DATA_SRC:-${HOME_REPO}/data}"
export VENV_PATH="${VENV_PATH:-$HOME/env_ema_tesi}"
MAIL="${MAIL:-nome.cognome@studenti.polito.it}"
# -----------------------------------------------------------------------------

export SWEEP_ID="${SWEEP_ID:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SCRATCH="${SCRATCH_FLASH}/ad_sweeps/${SWEEP_ID}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
STAGE="${STAGE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

# Exported to train_sweep.sbatch so both sides read the same grid.
export SWEEP_CATEGORIES="${SWEEP_CATEGORIES:-carpet reda_baseline reda_dustOnValidation reda_dustValidationAndTrain}"
export SWEEP_SEEDS="${SWEEP_SEEDS:-0 1 2 42 101}"
read -r -a CATEGORIES <<< "${SWEEP_CATEGORIES}"
read -r -a SEEDS      <<< "${SWEEP_SEEDS}"
N_RUNS=$(( ${#CATEGORIES[@]} * ${#SEEDS[@]} ))
# Ceiling division: a partially filled last node still needs a task.
N_TASKS=$(( (N_RUNS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))
ARRAY_SPEC="0-$(( N_TASKS - 1 ))%${MAX_PARALLEL}"

# Some clusters disable or severely limit job arrays. Detect it and fall back to
# N individual jobs rather than failing at submission time.
USE_ARRAY="${USE_ARRAY:-auto}"
if [[ "${USE_ARRAY}" == "auto" ]]; then
    MAX_ARRAY=$(scontrol show config 2>/dev/null \
                | awk -F= '/MaxArraySize/ {gsub(/[^0-9]/, "", $2); print $2}')
    if [[ -z "${MAX_ARRAY}" || "${MAX_ARRAY}" -lt "${N_TASKS}" ]]; then
        USE_ARRAY=0
    else
        USE_ARRAY=1
    fi
fi

# Each task runs 4 models in parallel, so walltime is the slowest single run
# plus contention, not 4x. gpu_a40 caps at 24 h.
PC_PARTITION="${PC_PARTITION:-gpu_a40}"
PC_TIME="${PC_TIME:-0-06:00:00}"
EAD_PARTITION="${EAD_PARTITION:-gpu_a40}"
EAD_TIME="${EAD_TIME:-0-23:00:00}"

if [[ ! -d "${DATA_SRC}" ]]; then
    echo "[error] data directory not found: ${DATA_SRC}" >&2
    exit 2
fi
MISSING=""
for C in "${CATEGORIES[@]}"; do
    [[ -d "${DATA_SRC}/${C}" ]] || MISSING="${MISSING} ${C}"
done
if [[ -n "${MISSING}" ]]; then
    echo "[error] categories missing under ${DATA_SRC}:${MISSING}" >&2
    echo "        available: $(ls -1 "${DATA_SRC}" | tr '\n' ' ')" >&2
    exit 2
fi

mkdir -p slurm_logs "sweeps/${SWEEP_ID}"

echo "sweep id      : ${SWEEP_ID}"
echo "repo (\$HOME)  : ${HOME_REPO}"
echo "scratch       : ${SWEEP_SCRATCH}"
echo "data src      : ${DATA_SRC}"
echo "venv          : ${VENV_PATH}"
echo "categories    : ${SWEEP_CATEGORIES}"
echo "seeds         : ${SWEEP_SEEDS}"
echo "grid          : ${#CATEGORIES[@]} x ${#SEEDS[@]} = ${N_RUNS} runs per model"
if [[ "${USE_ARRAY}" == "1" ]]; then
    echo "packing       : ${GPUS_PER_JOB} runs/node, array ${ARRAY_SPEC}"
else
    echo "packing       : ${GPUS_PER_JOB} runs/node, ${N_TASKS} single jobs (arrays unavailable)"
fi
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
        --export=ALL,HOME_REPO,DATA_SRC,SWEEP_SCRATCH,SWEEP_CATEGORIES \
        <<'STAGE_EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "[stage] target: ${SWEEP_SCRATCH}"
mkdir -p "${SWEEP_SCRATCH}/code" "${SWEEP_SCRATCH}/data" "${SWEEP_SCRATCH}/configs"

rsync -a --delete \
      --exclude '.git' --exclude 'sweeps' --exclude 'slurm_logs' \
      --exclude 'data' --exclude '__pycache__' --exclude 'results' \
      "${HOME_REPO}/" "${SWEEP_SCRATCH}/code/"

# One copy: categories and imagenette together.
rsync -a "${DATA_SRC}/" "${SWEEP_SCRATCH}/data/"

echo "[stage] staged under data/:"
ls -1 "${SWEEP_SCRATCH}/data"
du -sh "${SWEEP_SCRATCH}/data"

# ---- validate the whole grid before any GPU node is allocated ----
echo "[stage] validating the grid ..."
cd "${SWEEP_SCRATCH}/code"
BAD=0
for MODEL in patchcore efficientad; do
    for CATEGORY in ${SWEEP_CATEGORIES}; do
        python slurm/make_run_config.py --check-only \
            --model "${MODEL}" --category "${CATEGORY}" --seed 0 \
            --dataset-root "${SWEEP_SCRATCH}/data" \
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

EXPORTS="ALL,SWEEP_ID,HOME_REPO,GPUS_PER_JOB,SWEEP_CATEGORIES,SWEEP_SEEDS,VENV_PATH"

submit_model() {
    local model="$1" partition="$2" walltime="$3"
    local ids=() t
    if [[ "${USE_ARRAY}" == "1" ]]; then
        ids+=( "$(sbatch --parsable ${DEP} \
            --job-name="ad_${model}_${SWEEP_ID}" \
            --array="${ARRAY_SPEC}" \
            --partition="${partition}" --time="${walltime}" \
            --mail-user="${MAIL}" --export="${EXPORTS}" \
            slurm/train_sweep.sbatch "${model}")" )
    else
        for (( t = 0; t < N_TASKS; t++ )); do
            ids+=( "$(sbatch --parsable ${DEP} \
                --job-name="ad_${model}_${SWEEP_ID}_t${t}" \
                --partition="${partition}" --time="${walltime}" \
                --output="slurm_logs/%x_%j.out" --error="slurm_logs/%x_%j.err" \
                --mail-user="${MAIL}" --export="${EXPORTS},TASK_ID=${t}" \
                slurm/train_sweep.sbatch "${model}")" )
        done
    fi
    echo "${ids[@]}"
}

JID_PC=$(submit_model patchcore   "${PC_PARTITION}"  "${PC_TIME}")
echo "patchcore     : ${JID_PC}  (${N_TASKS} x ${GPUS_PER_JOB} runs, ${PC_PARTITION}, ${PC_TIME})"

JID_EAD=$(submit_model efficientad "${EAD_PARTITION}" "${EAD_TIME}")
echo "efficientad   : ${JID_EAD}  (${N_TASKS} x ${GPUS_PER_JOB} runs, ${EAD_PARTITION}, ${EAD_TIME})"

cat <<INFO

monitor    : squeue -u \$USER
per run    : grep RESULT slurm_logs/ad_*_${SWEEP_ID}_*.out
details    : sacct -j $(echo ${JID_PC} | tr " " ",") --format=JobID,State,Elapsed,MaxRSS,ExitCode
cancel all : scancel ${JID_PC} ${JID_EAD}

results land in ${HOME_REPO}/sweeps/${SWEEP_ID}/<model>/<category>/seed<N>/
aggregate with: python slurm/aggregate_results.py sweeps/${SWEEP_ID}
INFO
