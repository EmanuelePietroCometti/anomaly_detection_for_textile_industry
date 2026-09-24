#!/bin/bash
#SBATCH --job-name=ad_run
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s346291@studenti.polito.it
#
# 12 core e 96 GB = quota di un nodo gpu_a40 per GPU (48 core / 4 GPU, RAM = 2x VRAM).
# gpu_a40 ha un tetto di 24h: se le run non ci stanno, lancia un seed per volta.
#
# uso:
#   sbatch slurm/run_seeds_efficientad.sbatch <cartella_dataset> <categoria>        # 5 seed
#   sbatch slurm/run_seeds_efficientad.sbatch <cartella_dataset> <categoria> 42     # un seed solo
#
# <cartella_dataset> contiene le categorie: <cartella_dataset>/<categoria>/{train/good,ground_truth,...}
# ImageNette: di default <cartella_dataset>/imagenette_for_efficientad, altrimenti
#   IMAGENETTE_DIR=/percorso sbatch slurm/run_seeds_efficientad.sbatch ...
# Va scaricata una volta dal nodo di login (i nodi di calcolo non hanno rete).
# Output: result_<modello>/seed<seed>/<categoria>/
# Riprende: le run concluse lasciano un .done e vengono saltate.

set -uo pipefail          # NIENTE -e: un seed che fallisce non ferma gli altri

USAGE="uso: sbatch run_seeds.sbatch <cartella_dataset> <categoria> [seed ...]"
DATA_ROOT="${1:?${USAGE}}"
CATEGORY="${2:?${USAGE}}"
shift 2
if [[ $# -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(0 1 2 42 101); fi
MODELS=(efficientad)

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

# Percorso assoluto: un path relativo si intende rispetto alla dir da cui fai sbatch.
[[ -d "${DATA_ROOT}" ]] || { echo "[error] cartella dataset inesistente: ${DATA_ROOT}"; exit 2; }
DATA_ROOT="$(realpath "${DATA_ROOT}")"
IMAGENETTE_DIR="${IMAGENETTE_DIR:-${DATA_ROOT}/imagenette_for_efficientad}"

export PATH="${HOME}/env_ema_tesi/bin:${PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1        # i nodi di calcolo non hanno rete: usa la cache

# Pre-flight: meglio morire qui che dopo un'ora di training.
[[ -d "${DATA_ROOT}/${CATEGORY}/train/good" ]]   || { echo "[error] manca ${DATA_ROOT}/${CATEGORY}/train/good"; exit 2; }
[[ -d "${DATA_ROOT}/${CATEGORY}/ground_truth" ]] || { echo "[error] manca ${DATA_ROOT}/${CATEGORY}/ground_truth"; exit 2; }
[[ -d "pre_trained/efficientad_pretrained_weights" ]] || { echo "[error] pesi EfficientAD non pre-scaricati"; exit 2; }
[[ -d "${IMAGENETTE_DIR}" ]] || { echo "[error] manca ImageNette in ${IMAGENETTE_DIR} (scaricala dal nodo di login)"; exit 2; }
nvidia-smi -L | grep -q GPU              || { echo "[error] nessuna GPU: manca --gres?"; exit 3; }
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
                                         || { echo "[error] torch non vede la GPU"; exit 3; }

echo "nodo: $(hostname)  dataset: ${DATA_ROOT}  categoria: ${CATEGORY}  seed: ${SEEDS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SEED in "${SEEDS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    RUN_DIR="result_${MODEL}/seed${SEED}/${CATEGORY}"
    if [[ -f "${RUN_DIR}/.done" ]]; then
      echo "[skip] ${RUN_DIR}"
      continue
    fi
    mkdir -p "${RUN_DIR}"
    echo "===== ${CATEGORY} | seed ${SEED} | ${MODEL} | $(date -Is) ====="
    python main.py --baseline "${MODEL}" --seed "${SEED}" \
                   --data-root "${DATA_ROOT}" \
                   --imagenette-dir "${IMAGENETTE_DIR}" \
                   --category "${CATEGORY}" --run-dir "${RUN_DIR}" \
      2>&1 | tee "${RUN_DIR}/train.log"
    rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then
      touch "${RUN_DIR}/.done"
      echo "[OK] ${RUN_DIR}"
    else
      echo "[FAIL] ${RUN_DIR} exit=${rc}"
    fi
  done
done

echo "finito: $(date -Is)"
