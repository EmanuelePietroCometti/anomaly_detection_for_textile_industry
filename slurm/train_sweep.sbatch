#!/usr/bin/env bash
# =============================================================================
# Anomaly detection sweep on HPC@PoliTO Legion - full-node A40 packing.
#
# 4 categories x 5 seeds = 20 runs per model, 4 runs per job (one per GPU)
# => 5 array tasks per model. Each task holds a whole gpu_a40 node.
#
# Do NOT submit by hand: submit_sweep.sh stages code and dataset onto
# $SCRATCH_FLASH, validates the grid, and chains this with --dependency.
#
# Run list order (categories outer, seeds inner):
#   task 0 -> carpet/0  carpet/1  carpet/2  carpet/42
#   task 1 -> carpet/101  no_dust/0  no_dust/1  no_dust/2
#   task 2 -> no_dust/42  no_dust/101  dustOnValidation/0  dustOnValidation/1
#   task 3 -> dustOnValidation/2  /42  /101  dustOnValidationAndTrain/0
#   task 4 -> dustOnValidationAndTrain/1  /2  /42  /101
# =============================================================================
#SBATCH --job-name=ad_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=384G
#SBATCH --gres=gpu:4
#SBATCH --time=0-23:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nome.cognome@studenti.polito.it     # <-- CAMBIA
# -----------------------------------------------------------------------------
# A whole gpu_a40 node: 4 GPUs, 4 tasks (1 per GPU), 12 cores per task = 48
# cores, 384 GB = 4 x (2 x 48 GB VRAM). This is the guide's ratio applied to the
# full node. The 4 runs are independent Python processes, not MPI ranks, so they
# are launched as background subshells rather than through srun.
# -----------------------------------------------------------------------------

set -uo pipefail

# ============================= sweep definition ==============================
MODEL="${1:-}"
if [[ "${MODEL}" != "patchcore" && "${MODEL}" != "efficientad" ]]; then
    echo "usage: sbatch train_sweep.sbatch {patchcore|efficientad}" >&2
    exit 2
fi

CATEGORIES=(carpet no_dust dustOnValidation dustOnValidationAndTrain)
SEEDS=(0 1 2 42 101)
GPUS_PER_JOB="${GPUS_PER_JOB:-4}"

RUNS=()
for C in "${CATEGORIES[@]}"; do
    for S in "${SEEDS[@]}"; do
        RUNS+=("${C}:${S}")
    done
done

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# ============================ paths and staging ==============================
: "${SCRATCH_FLASH:?SCRATCH_FLASH is not set - are you on a Legion node?}"
: "${SWEEP_ID:?SWEEP_ID is not set - submit through submit_sweep.sh}"

HOME_REPO="${HOME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SWEEP_SCRATCH="${SCRATCH_FLASH}/ad_sweeps/${SWEEP_ID}"
CODE_DIR="${SWEEP_SCRATCH}/code"
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
source "${HOME}/venvs/anomalib/bin/activate"
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Shared by all 4 processes, kept off $HOME: caches are hot data, the NAS is slow.
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
    echo "[error] ${GPUS_PER_JOB} GPUs requested, ${N_GPU} visible - check --gres=gpu:4" >&2
    exit 4
fi

cd "${CODE_DIR}"

# ============================= single run body ===============================
run_one() {
    local slot="$1" category="$2" seed="$3"
    local run_dir="${SWEEP_SCRATCH}/runs/${MODEL}/${category}/seed${seed}"
    local run_config="${SWEEP_SCRATCH}/configs/${MODEL}_${category}_seed${seed}.yaml"
    local home_run_dir="${HOME_REPO}/sweeps/${SWEEP_ID}/${MODEL}/${category}/seed${seed}"

    # One GPU per process: main.py and the anomalib Engine stay single-device,
    # no code change needed.
    export CUDA_VISIBLE_DEVICES="${slot}"

    # Stagger the starts so 4 processes don't hit the dataset and build their
    # dataloaders at the same instant.
    sleep $(( slot * 15 ))

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

    # Sync this run home as soon as it finishes, not at the end of the job:
    # if a sibling run later hits the walltime, these results are already safe.
    mkdir -p "${home_run_dir}"
    rsync -a "${run_dir}/" "${home_run_dir}/"
    cp "${run_config}" "${home_run_dir}/run_config.yaml" 2>/dev/null || true

    echo "[slot ${slot}] END   ${category}/seed${seed} exit=${status} at $(date -Is)"
    return ${status}
}

# ============================ launch the 4 runs ==============================
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
