#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#
# Usage: rename_and_verify_wheels.sh
# This script is aware of TFCI_ variables, so it doesn't need any arguments.
# Puts new wheel through auditwheel to rename and verify it, deletes the old
# one, checks the filesize, and then ensures the new wheel is installable.
set -exo pipefail

cd "$TFCI_OUTPUT_DIR"

# Install UMD compat library if enabled
if [[ "$TFCI_BAZEL_HERMETIC_CUDA_UMD_ENABLE" == 1 ]]; then
  # Extract the UMD version resolved for this build directly from .bazelrc
  export HERMETIC_CUDA_UMD_VERSION=""
  TEST_CONFIG="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_wheel_test"
  CONFIG_LINE=$(grep "^test:${TEST_CONFIG} " "$TFCI_GIT_DIR/.bazelrc" || true)
  if [[ "$CONFIG_LINE" =~ HERMETIC_CUDA_UMD_VERSION=\"?([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    export HERMETIC_CUDA_UMD_VERSION="${BASH_REMATCH[1]}"
  else
    for conf in $(echo "$CONFIG_LINE" | grep -o -e '--config=[a-zA-Z0-9_-]*' | sed 's/--config=//'); do
      SUBCONFIG_LINE=$(grep "^test:${conf} " "$TFCI_GIT_DIR/.bazelrc" || true)
      if [[ "$SUBCONFIG_LINE" =~ HERMETIC_CUDA_UMD_VERSION=\"?([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        export HERMETIC_CUDA_UMD_VERSION="${BASH_REMATCH[1]}"
        break
      fi
    done
  fi

  if [[ -n "$HERMETIC_CUDA_UMD_VERSION" ]]; then
    echo "Installing UMD compat library inside the container..."
    if [[ "$HERMETIC_CUDA_UMD_VERSION" =~ ^([0-9]+)\.([0-9]+) ]]; then
      COMPAT_VERSION="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
    else
      echo "Error: Invalid HERMETIC_CUDA_UMD_VERSION format ($HERMETIC_CUDA_UMD_VERSION)."
      exit 1
    fi

    echo "Setting up NVIDIA apt repository..."
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub || true
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia.list

    echo "Running apt-get to install cuda-compat-${COMPAT_VERSION}..."
    apt-get update -y

    # Fetch the exact UMD version that Bazel will download from NVIDIA redistrib JSON.
    BAZEL_UMD_VERSION=$(curl -s "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_${HERMETIC_CUDA_UMD_VERSION}.json" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("nvidia_driver", {}).get("version", ""))')

    if [[ -z "$BAZEL_UMD_VERSION" ]]; then
      echo "Warning: Could not extract Bazel UMD version from redistrib JSON. Falling back to default cuda-compat-${COMPAT_VERSION}."
      apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}"
    else
      echo "Hermetic UMD version from Bazel redistrib json: $BAZEL_UMD_VERSION"
      # Find the exact package version in apt-cache that matches the Bazel UMD version.
      EXACT_PKG=$(apt-cache madison cuda-compat-${COMPAT_VERSION} | grep "$BAZEL_UMD_VERSION" | head -n1 | cut -d"|" -f2 | tr -d " " || true)

      if [[ -n "$EXACT_PKG" ]]; then
        echo "Found exact match for Bazel UMD version $BAZEL_UMD_VERSION: $EXACT_PKG"
        apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}=${EXACT_PKG}"
      else
        echo "Warning: Could not find exact apt package match for $BAZEL_UMD_VERSION. Falling back to default cuda-compat-${COMPAT_VERSION}."
        apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}"
      fi
    fi

    COMPAT_DIR="/usr/local/cuda-${COMPAT_VERSION/-/.}/compat"
    # Clean up any system-wide ldconfig registration to avoid affecting other tasks like Bazel tests.
    # Instead, we rely on LD_LIBRARY_PATH dynamic configuration below.
    rm -f /etc/ld.so.conf.d/cuda-compat.conf
    ldconfig
    echo "Successfully installed and configured cuda-compat-${COMPAT_VERSION}."
  fi
fi

# Move extra wheel files somewhere out of the way. This script
# expects just one wheel file to exist.
if [[ "$(ls *.whl | wc -l | tr -d ' ')" != "1" ]]; then
  echo "More than one wheel file is present: moving the oldest to"
  echo "$TFCI_OUTPUT_DIR/extra_wheels."
  # List all .whl files by their modification time (ls -t) and move anything
  # other than the most recently-modified one (the newest one).
  mkdir -p $TFCI_OUTPUT_DIR/extra_wheels
  ls -t *.whl | tail -n +2 | xargs mv -t $TFCI_OUTPUT_DIR/extra_wheels
fi

# Repair wheels with auditwheel and delete the old one.
if [[ "$TFCI_WHL_AUDIT_ENABLE" == "1" ]]; then
  python3 -m auditwheel repair --plat "$TFCI_WHL_AUDIT_PLAT" --wheel-dir . *.whl
  # if the wheel is already named correctly, auditwheel won't rename it. so we
  # list all .whl files by their modification time (ls -t) and delete anything
  # other than the most recently-modified one (the new one).
  ls -t *.whl | tail -n +2 | xargs rm
fi

# Check if size is too big. TFCI_WHL_SIZE_LIMIT is in find's format, which can be
# 'k' for kilobytes, 'M' for megabytes, or 'G' for gigabytes, and the + to indicate
# "anything greater than" is added by the script.
if [[ "$TFCI_WHL_SIZE_LIMIT_ENABLE" == "1" ]] && [[ -n "$("$TFCI_FIND_BIN" . -iname "*.whl" -size "+$TFCI_WHL_SIZE_LIMIT")" ]]; then
  echo "Error: Generated wheel is too big! Limit is $TFCI_WHL_SIZE_LIMIT"
  echo '(search for TFCI_WHL_SIZE_LIMIT to change it)'
  ls -sh *.whl
  exit 2
fi

# Quick install checks
venv_dir=$(mktemp -d)
if [[ $(uname -s) != MSYS_NT* ]]; then
  "python${TFCI_PYTHON_VERSION}" -m venv "$venv_dir"
  python="$venv_dir/bin/python3"
else
  # When using the Linux-like path, venv creation quietly fails, which is
  # why it's converted here.
  venv_dir=$(cygpath -m $venv_dir)
  "/c/python${TFCI_PYTHON_VERSION}/python.exe" -m venv "$venv_dir"
  python="$venv_dir/Scripts/python.exe"
fi

# TODO(b/366266944) Remove the check after tf docker image upgrade for NumPy 2
# and numpy 1 support is dropped b/361369076.
if [[ "$TFCI_WHL_NUMPY_VERSION" == 1 ]]; then
  if [[ "$TFCI_PYTHON_VERSION" == "3.13" ]]; then
    "$python" -m pip install numpy==1.26.4
  else
    "$python" -m pip install numpy==1.26.0
  fi
fi


if [[ "$TFCI_BAZEL_COMMON_ARGS" =~ gpu|cuda ]]; then
  echo "Checking to make sure tensorflow[and-cuda] is installable..."
  "$python" -m pip install "$(echo *.whl)[and-cuda]" $TFCI_PYTHON_VERIFY_PIP_INSTALL_ARGS
else
  "$python" -m pip install *.whl $TFCI_PYTHON_VERIFY_PIP_INSTALL_ARGS
fi

# Detect versioned or unversioned compat library paths for LD_LIBRARY_PATH
for dir in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
  if [[ -d "$dir" ]]; then
    export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH:-}"
    break
  fi
done

if [[ "$TFCI_WHL_IMPORT_TEST_ENABLE" == "1" ]]; then
  if [[ "$TFCI_BAZEL_COMMON_ARGS" =~ gpu|cuda ]]; then
    "$python" -c '
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
  raise ValueError("No GPU devices found!")
print(f"Successfully found GPU devices: {gpus}")
t1=tf.constant([1,2,3,4])
t2=tf.constant([5,6,7,8])
print(tf.add(t1,t2).shape)
'
  else
    "$python" -c '
import tensorflow as tf
t1=tf.constant([1,2,3,4])
t2=tf.constant([5,6,7,8])
print(tf.add(t1,t2).shape)
'
  fi
  "$python" -c '
import sys
import tensorflow as tf
sys.exit(0 if "keras" in tf.keras.__name__ else 1)
'
fi

# Import tf nightly wheel built with numpy2 from PyPI in numpy1 env for testing.
# This aims to maintain TF compatibility with NumPy 1.x until 2025 b/361369076.
if [[ "$TFCI_WHL_NUMPY_VERSION" == 1 ]]; then
  # Uninstall tf nightly wheel built with numpy1.
  "$python" -m pip uninstall -y tf_nightly_numpy1
  # Install tf nightly cpu wheel built with numpy2.x from PyPI in numpy1.x env.
  "$python" -m pip install tf-nightly-cpu
  if [[ "$TFCI_WHL_IMPORT_TEST_ENABLE" == "1" ]]; then
    "$python" -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
    "$python" -c 'import sys; import tensorflow as tf; sys.exit(0 if "keras" in tf.keras.__name__ else 1)'
  fi
fi
