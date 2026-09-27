#!/bin/bash
# Copyright 2026 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
VENV_DIR="${1:-${VENV_DIR:-${SCRIPT_DIR}/.venv}}"

echo "============================================================"
echo "Setting up environment for OpenXLA TPU microbenchmarks"
echo "Repository root: ${REPO_ROOT}"
echo "Virtual environment directory: ${VENV_DIR}"
echo "============================================================"

# Minimum Python version required (JAX 0.11+ requires Python >= 3.12)
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=12

# Find suitable python binary
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3.13 python3; do
    if command -v "${candidate}" &> /dev/null; then
      if "${candidate}" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) else 1)" 2>/dev/null; then
        PYTHON_BIN="${candidate}"
        break
      fi
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" &> /dev/null; then
  echo "Error: Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required (JAX 0.11+ requires Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR})." >&2
  echo "Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or higher, or specify PYTHON=/path/to/python in the environment." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) else 1)" 2>/dev/null; then
  CURRENT_PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
  echo "Error: Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required, but ${PYTHON_BIN} is ${CURRENT_PY_VER}." >&2
  echo "JAX 0.11+ requires Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}." >&2
  echo "Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or higher, or specify PYTHON=/path/to/python in the environment." >&2
  exit 1
fi

echo "Using Python: $("${PYTHON_BIN}" --version) (${PYTHON_BIN})"

# Create virtual environment if it does not exist, or recreate if incompatible
if [[ -d "${VENV_DIR}" ]]; then
  if [[ -f "${VENV_DIR}/bin/python3" ]]; then
    if ! "${VENV_DIR}/bin/python3" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) else 1)" 2>/dev/null; then
      EXISTING_PY_VER="$("${VENV_DIR}/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 'unknown')"
      echo "Warning: Existing virtual environment at ${VENV_DIR} uses Python ${EXISTING_PY_VER}, but Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required."
      echo "Recreating virtual environment with $("${PYTHON_BIN}" --version)..."
      rm -rf "${VENV_DIR}"
    else
      echo "Using existing virtual environment at ${VENV_DIR}."
    fi
  else
    echo "Virtual environment at ${VENV_DIR} appears incomplete. Recreating..."
    rm -rf "${VENV_DIR}"
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing requirements from ${SCRIPT_DIR}/requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Configure site-packages .pth file so xla is importable from anywhere
echo "Configuring Python site-packages path for OpenXLA repository..."
python3 -c "
import site, pathlib, sys
site_dirs = site.getsitepackages()
if site_dirs:
    pth_file = pathlib.Path(site_dirs[0]) / 'openxla_benchmarks.pth'
    pth_file.write_text(sys.argv[1] + '\n')
" "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "Setup completed successfully!"
echo ""
echo "To activate this virtual environment in your shell, run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run the benchmarks suite, run:"
echo "  python3 ${SCRIPT_DIR}/run_benchmarks.py"
echo "============================================================"
