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

# Check for python3
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 is not installed or not found in PATH." >&2
  exit 1
fi

# Create virtual environment if it does not already exist
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
else
  echo "Using existing virtual environment at ${VENV_DIR}."
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
