#!/usr/bin/env bash
# =============================================================================
# Anomaly detection sweep on HPC@PoliTO Legion - 1 run per job.
#
# 4 categories x 5 seeds = 20 runs per model, 1 run per job
# => 20 array tasks per model.
#
# Do NOT submit by hand: submit_sweep.sh stages code and dataset onto
# $SCRATCH_FLASH, validates the grid, and chains this with --dependency.
# =============================================================================
#SBATCH --job-name=ad_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
# NOTE: no #SBATCH --array directive here on purpose. Legion has
# MaxArraySize=0 (job arrays disabled), so a directive in this file would
# be rejected at submission even when submit_sweep.sh does not want an
# array. submit_sweep.sh passes --array on the command line when the
# cluster supports it, and TASK_ID otherwise.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s346291@studenti.polito.it
# -----------------------------------------------------------------------------
# 1 GPU, 1 task, 12 cores, 48 GB RAM.
# This follows the HPC@PoliTO guide ratio for gpu_a40 (48 cores / 4 GPUs = 12).
# -----------------------------------------------------------------------------

set -uo pipefail

# ============================= sweep definition ==============================
MODEL="${1:-}"
if [[ "${MODEL}" != "patchcore" && "${MODEL}" != "efficientad" ]]; then
    echo "usage: sbatch train_sweep.sbatch {patchcore|efficientad}" >&2
    exit 2
fi

# Single source of truth: submit_sweep.sh exports these as space-separated
# strings, so the grid here can never drift from the one used to size the array.
read -r -a CATEGORIES <<< "${SWEEP_CATEGORIES:-carpet reda_baseline reda_dustOnValidation reda_dustValidationAndTrain}"
read -r -a SEEDS      <<< "${SWEEP_SEEDS:-0 1 2 42 101}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

RUNS=()
for C in "${CATEGORIES[@]}"; do
    for S in "${SEEDS[@]}"; do
        RUNS+=("${C}:${S}")
    done
done

# SLURM_ARRAY_TASK_ID inside a job array; TASK_ID when arrays are
# unavailable and submit_sweep.sh falls back to individual jobs.
IDX="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

# ============================ paths and staging ==============================
: "${SCRATCH_FLASH:?SCRATCH_FLASH is not set - are you on a Legion node?}"
: "${SWEEP_ID:?SWEEP_ID is not set - submit through submit_sweep.sh}"

HOME_REPO="${HOME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SWEEP_SCRATCH="${SCRATCH_FLASH}/ad_sweeps/${SWEEP_ID}"
CODE_DIR="${SWEEP_SCRATCH}/code"
# The repo keeps categories directly under data/, with imagenette alongside
# them (it is not a category, the grid never looks at it).
DATA_DIR="${SWEEP_SCRATCH}/data"
IMAGENETTE_DIR="${SWEEP_SCRATCH}/data/imagenette_for_efficientad"

if [[ ! -d "${CODE_DIR}" || ! -d "${DATA_DIR}" ]]; then
    echo "[error] staging missing under ${SWEEP_SCRATCH}." >&2
    echo "        run ./slurm/submit_sweep.sh instead of submitting by hand." >&2
    exit 3
fi

# ============================== environment ==================================
# --- ADATTA AL TUO AMBIENTE --------------------------------------------------
module purge
module load nvhpc-hpcx-cuda12/25.1
source "${VENV_PATH:-${HOME}/env_ema_tesi}/bin/activate"
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Kept off $HOME: caches are hot data, the NAS is slow.
export TORCH_HOME="${SWEEP_SCRATCH}/cache/torch"
export XDG_CACHE_HOME="${SWEEP_SCRATCH}/cache"
mkdir -p "${TORCH_HOME}"

echo "=============================================================="
echo "job      : ${SLURM_JOB_ID:-local}  (array ${SLURM_ARRAY_JOB_ID:-?} task ${IDX})"
echo "node     : $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}"
echo "model    : ${MODEL}"
echo "packing  : ${GPUS_PER_JOB} runs in parallel, one per GPU"
echo "started  : $(date -Is)"
echo "=============================================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
N_GPU=$(nvidia-smi -L | grep -c GPU || true)
if [[ "${N_GPU}" -lt "${GPUS_PER_JOB}" ]]; then
    echo "[error] ${GPUS_PER_JOB} GPUs requested, ${N_GPU} visible - check --gres=gpu:${GPUS_PER_JOB}" >&2
    exit 4
fi

cd "${CODE_DIR}"

# ============================= single run body ===============================
run_one() {
    local slot="$1" category="$2" seed="$3"
    local run_dir="${SWEEP_SCRATCH}/runs/${MODEL}/${category}/seed${seed}"
    local run_config="${SWEEP_SCRATCH}/configs/${MODEL}_${category}_seed${seed}.yaml"
    local home_run_dir="${HOME_REPO}/sweeps/${SWEEP_ID}/${MODEL}/${category}/seed${seed}"

    export CUDA_VISIBLE_DEVICES="${slot}"

    mkdir -p "${run_dir}"
    echo "[slot ${slot}] START ${category}/seed${seed} on GPU ${slot} at $(date -Is)"

    python slurm/make_run_config.py \
        --template        config.yaml \
        --model           "${MODEL}" \
        --category        "${category}" \
        --seed            "${seed}" \
        --dataset-root    "${DATA_DIR}" \
        --imagenette-dir  "${IMAGENETTE_DIR}" \
        --run-dir         "${run_dir}" \
        --out             "${run_config}" \
        --num-workers     "${SLURM_CPUS_PER_TASK:-12}" \
        --gpu-slot        "${slot}" \
        --slurm-job-id    "${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
        >> "${run_dir}/train.log" 2>&1
    local cfg_status=$?
    if [[ ${cfg_status} -ne 0 ]]; then
        echo "[slot ${slot}] config/pre-flight FAILED for ${category}/seed${seed}"
        return ${cfg_status}
    fi

    AD_CONFIG="${run_config}" python main.py \
        --baseline "${MODEL}" --seed "${seed}" \
        >> "${run_dir}/train.log" 2>&1
    local status=$?

    # Sync this run home as soon as it finishes, not at the end of the job
    mkdir -p "${home_run_dir}"
    rsync -a "${run_dir}/" "${home_run_dir}/"
    cp "${run_config}" "${home_run_dir}/run_config.yaml" 2>/dev/null || true

    echo "[slot ${slot}] END   ${category}/seed${seed} exit=${status} at $(date -Is)"
    return ${status}
}

# ============================ launch the run =================================
PIDS=()
LABELS=()

for SLOT in $(seq 0 $((GPUS_PER_JOB - 1))); do
    ENTRY_IDX=$(( IDX * GPUS_PER_JOB + SLOT ))
    if [[ ${ENTRY_IDX} -ge ${#RUNS[@]} ]]; then
        echo "[slot ${SLOT}] no run assigned (grid exhausted)"
        continue
    fi
    ENTRY="${RUNS[${ENTRY_IDX}]}"
    CATEGORY="${ENTRY%%:*}"
    SEED="${ENTRY##*:}"

    run_one "${SLOT}" "${CATEGORY}" "${SEED}" &
    PIDS+=($!)
    LABELS+=("${CATEGORY}/seed${SEED}")
done

# ========================= wait and report per run ===========================
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[RESULT] OK    ${MODEL}  ${LABELS[$i]}"
    else
        echo "[RESULT] FAIL  ${MODEL}  ${LABELS[$i]}"
        FAILED=1
    fi
done

echo "=============================================================="
echo "finished : $(date -Is)   task exit=${FAILED}"
echo "logs per run: ${SWEEP_SCRATCH}/runs/${MODEL}/<category>/seed<N>/train.log"
echo "=============================================================="
exit "${FAILED}"