#!/usr/bin/env bash
# =============================================================================
# Anomaly detection sweep on HPC@PoliTO Legion
# 4 categories x 5 seeds = 20 array tasks, one model per submission.
#
# Do NOT submit this by hand: submit_sweep.sh stages code and dataset onto
# $SCRATCH_FLASH first and chains this job to that staging with --dependency.
#
# Array index -> (category, seed):
#   idx  0..4   carpet                    seeds 0,1,2,42,101
#   idx  5..9   no_dust                   seeds 0,1,2,42,101
#   idx 10..14  dustOnValidation          seeds 0,1,2,42,101
#   idx 15..19  dustOnValidationAndTrain  seeds 0,1,2,42,101
# =============================================================================
#SBATCH --job-name=ad_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-19%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-20:00:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s346291@studenti.polito.it
# -----------------------------------------------------------------------------
# cpus-per-task=12 and mem=96G follow the guide's ratio for gpu_a40:
# 12 cores per GPU, RAM = 2 x VRAM = 2 x 48 GB per GPU. Requesting more would
# block cores and memory other jobs could use on the same node.
# -----------------------------------------------------------------------------

set -euo pipefail

# ============================= sweep definition ==============================
MODEL="${1:-}"
if [[ "${MODEL}" != "patchcore" && "${MODEL}" != "efficientad" ]]; then
    echo "usage: sbatch train_sweep.sbatch {patchcore|efficientad}" >&2
    exit 2
fi

CATEGORIES=(carpet no_dust dustOnValidation dustOnValidationAndTrain)
SEEDS=(0 1 2 42 101)

N_SEEDS=${#SEEDS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CATEGORY="${CATEGORIES[$((IDX / N_SEEDS))]}"
SEED="${SEEDS[$((IDX % N_SEEDS))]}"

# ============================ paths and staging ==============================
: "${SCRATCH_FLASH:?SCRATCH_FLASH is not set - are you on a Legion node?}"
: "${SWEEP_ID:?SWEEP_ID is not set - submit through submit_sweep.sh}"

HOME_REPO="${HOME_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SWEEP_SCRATCH="${SCRATCH_FLASH}/ad_sweeps/${SWEEP_ID}"
CODE_DIR="${SWEEP_SCRATCH}/code"
DATA_DIR="${SWEEP_SCRATCH}/data/mvtec"
IMAGENETTE_DIR="${SWEEP_SCRATCH}/data/imagenette_for_efficientad"

RUN_DIR="${SWEEP_SCRATCH}/runs/${MODEL}/${CATEGORY}/seed${SEED}"
RUN_CONFIG="${SWEEP_SCRATCH}/configs/${MODEL}_${CATEGORY}_seed${SEED}.yaml"
HOME_RUN_DIR="${HOME_REPO}/sweeps/${SWEEP_ID}/${MODEL}/${CATEGORY}/seed${SEED}"

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
# Keep torch/HF caches off $HOME: they are hot data and the NAS is slow.
export TORCH_HOME="${SWEEP_SCRATCH}/cache/torch"
export XDG_CACHE_HOME="${SWEEP_SCRATCH}/cache"
mkdir -p "${TORCH_HOME}"

echo "=============================================================="
echo "job       : ${SLURM_JOB_ID:-local}  (array ${SLURM_ARRAY_JOB_ID:-?} task ${IDX})"
echo "node      : $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}"
echo "model     : ${MODEL}"
echo "category  : ${CATEGORY}"
echo "seed      : ${SEED}"
echo "scratch   : ${RUN_DIR}"
echo "home copy : ${HOME_RUN_DIR}"
echo "started   : $(date -Is)"
echo "=============================================================="

# The guide is explicit: without --gres you get a GPU node but no GPU.
# Fail here rather than after an hour of silent CPU training.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! nvidia-smi -L | grep -q GPU; then
    echo "[error] no GPU visible - check --gres=gpu:1" >&2
    exit 4
fi

# =========================== per-run configuration ===========================
cd "${CODE_DIR}"
mkdir -p "${RUN_DIR}"

python slurm/make_run_config.py \
    --template        config.yaml \
    --model           "${MODEL}" \
    --category        "${CATEGORY}" \
    --seed            "${SEED}" \
    --dataset-root    "${DATA_DIR}" \
    --imagenette-dir  "${IMAGENETTE_DIR}" \
    --run-dir         "${RUN_DIR}" \
    --out             "${RUN_CONFIG}" \
    --num-workers     "${SLURM_CPUS_PER_TASK:-8}" \
    --slurm-job-id    "${SLURM_ARRAY_JOB_ID:-local}_${IDX}"

export AD_CONFIG="${RUN_CONFIG}"

# ================================ training ===================================
set +e
python main.py --baseline "${MODEL}" --seed "${SEED}" 2>&1 | tee "${RUN_DIR}/train.log"
STATUS=${PIPESTATUS[0]}
set -e

# ======================= copy results back to $HOME ==========================
# $SCRATCH_FLASH deletes anything older than 30 days, without notice.
# Results are small (reports, curves, checkpoints, exports), so they go home now.
echo "[INFO] syncing results to \$HOME ..."
mkdir -p "${HOME_RUN_DIR}"
rsync -a "${RUN_DIR}/" "${HOME_RUN_DIR}/"
cp "${RUN_CONFIG}" "${HOME_RUN_DIR}/run_config.yaml" 2>/dev/null || true

echo "=============================================================="
echo "finished : $(date -Is)   exit=${STATUS}"
echo "=============================================================="
exit "${STATUS}"
