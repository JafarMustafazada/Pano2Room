#!/usr/bin/env -S conda run --no-capture-output -n menv2 /bin/bash
# set -euo pipefail

PYTHON="$(command -v python)"

# ---------- workspace paths ----------
WORKDIR="${HOME}/workspace"
PROJECT_DIR="${WORKDIR}/Pano2Room"
SAVES_DIR="${WORKDIR}/saves"
CONDA_DIR=/opt/miniforge3
ENV_NAME=menv2

echo "version 2.3"
conda --version
echo "Workspace: ${WORKDIR}"
ls -la "${WORKDIR}" || true
echo "CUDA_HOME: ${CUDA_HOME}"

echo ""
echo "============================================"
echo "Environment ready!"
echo "Using python: ${PYTHON}"
echo "Pip: $(${PYTHON} -m pip --version 2>&1 | head -n1)"
echo "PyTorch: $(${PYTHON} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "CUDA available: $(${PYTHON} -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo "============================================"
echo ""

# ---------- run the project script ----------
cd "${PROJECT_DIR}"
echo "Running: sh scripts/run_Pano2Room.sh"
echo "Starting at: $(date)"
echo ""

# ${PYTHON} -m pip install line_profiler
# exec sh scripts/run_Pano2Room_profiler.sh

# exec sh scripts/run_Pano2Room.sh

python3 save_pc_as_mesh.py

echo ""
echo "Completed at: $(date)"
