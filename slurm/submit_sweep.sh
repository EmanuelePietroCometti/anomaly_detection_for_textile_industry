#!/usr/bin/env bash
# =============================================================================
# Submits the full sweep on HPC@PoliTO Legion:
#   1. a staging job (cpu_sapphire) that copies code + dataset to $SCRATCH_FLASH
#   2. the PatchCore array   (20 tasks), chained with --dependency=afterok
#   3. the EfficientAD array (20 tasks), same dependency
#
# Staging runs as a SLURM job, not on the login node: the guide requires large
# data transfers to go through the scheduler.
#
#   ./slurm/submit_sweep.sh
#   DATASET_SRC=$HOME/datasets/mvtec ./slurm/submit_sweep.sh
#   STAGE=0 SWEEP_ID=ablation_dust ./slurm/submit_sweep.sh   # scratch already staged
# =============================================================================

set -euo pipefail

: "${SCRATCH_FLASH:?SCRATCH_FLASH is not set - log in to Legion first}"

# --- ADATTA AI TUOI PERCORSI -------------------------------------------------
export HOME_REPO="${HOME_REPO:-$PWD}"
export DATASET_SRC="${DATASET_SRC:-$HOME_REPO/data}"
export IMAGENETTE_SRC="${IMAGENETTE_SRC:-$HOME_REPO/data/imagenette_for_efficientad}"
ARRAY="${ARRAY:-0-19%4}"
MAIL="${MAIL:-s346291@studenti.polito.it}"
# -----------------------------------------------------------------------------

export SWEEP_ID="${SWEEP_ID:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SCRATCH="${SCRATCH_FLASH}/ad_sweeps/${SWEEP_ID}"
STAGE="${STAGE:-1}"

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

# Code: small, excludes git history, previous sweeps and any local dataset copy.
rsync -a --delete \
      --exclude '.git' --exclude 'sweeps' --exclude 'slurm_logs' \
      --exclude 'data' --exclude '__pycache__' --exclude 'results' \
      "${HOME_REPO}/" "${SWEEP_SCRATCH}/code/"

rsync -a --exclude 'imagenette_for_efficientad' --exclude 'dataset_retraining' \
      "${DATASET_SRC}/" "${SWEEP_SCRATCH}/data/mvtec/"

# cache pesi pre-popolata sul nodo di login (i nodi di calcolo non hanno rete)
mkdir -p "${SWEEP_SCRATCH}/cache"
rsync -a "${HOME}/.cache/torch" "${HOME}/.cache/huggingface" "${SWEEP_SCRATCH}/cache/" || true

if [[ -d "${IMAGENETTE_SRC}" ]]; then
    rsync -a "${IMAGENETTE_SRC}/" "${SWEEP_SCRATCH}/data/imagenette_for_efficientad/"
else
    echo "[stage] WARNING: ${IMAGENETTE_SRC} not found - EfficientAD tasks will fail pre-flight"
fi

echo "[stage] categories staged:"
ls -1 "${SWEEP_SCRATCH}/data/mvtec"
du -sh "${SWEEP_SCRATCH}/data"
echo "[stage] done"
STAGE_EOF
)
    echo "staging       : job ${JID_STAGE} (cpu_sapphire)"
    DEP="--dependency=afterok:${JID_STAGE}"
else
    echo "staging       : skipped (STAGE=0)"
fi

# PatchCore: 1 epoch, coreset fit. EfficientAD: 100 epochs at batch size 1.
# gpu_a40 caps walltime at 24 h, so both stay under it.
JID_PC=$(sbatch --parsable ${DEP} \
    --job-name="ad_patchcore_${SWEEP_ID}" \
    --time=0-03:00:00 \
    --mail-user="${MAIL}" \
    --export=ALL,SWEEP_ID,HOME_REPO \
    slurm/train_sweep.sbatch patchcore)
echo "patchcore     : array ${JID_PC} (20 tasks, 3 h each)"

JID_EAD=$(sbatch --parsable ${DEP} \
    --job-name="ad_efficientad_${SWEEP_ID}" \
    --time=0-20:00:00 \
    --mail-user="${MAIL}" \
    --export=ALL,SWEEP_ID,HOME_REPO \
    slurm/train_sweep.sbatch efficientad)
echo "efficientad   : array ${JID_EAD} (20 tasks, 20 h each)"

cat <<INFO

monitor    : squeue -u \$USER
details    : sacct -j ${JID_PC} --format=JobID,State,Elapsed,MaxRSS,ExitCode
cancel all : scancel ${JID_PC} ${JID_EAD}

results are written on \$SCRATCH_FLASH and rsynced back to
  ${HOME_REPO}/sweeps/${SWEEP_ID}/<model>/<category>/seed<N>/

then aggregate with:
  python slurm/aggregate_results.py sweeps/${SWEEP_ID}
INFO
