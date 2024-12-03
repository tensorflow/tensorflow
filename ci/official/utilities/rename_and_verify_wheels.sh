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
set -euxo pipefail

cd "$TFCI_OUTPUT_DIR"

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
if [[ "$TFCI_WHL_SIZE_LIMIT_ENABLE" == "1" ]] && [[ -n "$(find . -iname "*.whl" -size "+$TFCI_WHL_SIZE_LIMIT")" ]]; then
  echo "Error: Generated wheel is too big! Limit is $TFCI_WHL_SIZE_LIMIT"
  echo '(search for TFCI_WHL_SIZE_LIMIT to change it)'
  ls -sh *.whl
  exit 2
fi

# Quick install checks
venv=$(mktemp -d)
"python${TFCI_PYTHON_VERSION}" -m venv "$venv"
python="$venv/bin/python3"
# TODO(b/366266944) Remove the check after tf docker image upgrade for NumPy 2
# and numpy 1 support is dropped b/361369076.
if [[ "$TFCI_WHL_NUMPY_VERSION" == 1 ]]; then
  "$python" -m pip install numpy==1.26.0
fi
"$python" -m pip install *.whl $TFCI_PYTHON_VERIFY_PIP_INSTALL_ARGS
if [[ "$TFCI_WHL_IMPORT_TEST_ENABLE" == "1" ]]; then
  "$python" -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
  "$python" -c 'import sys; import tensorflow as tf; sys.exit(0 if "keras" in tf.keras.__name__ else 1)'
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
# VERY basic check to ensure the [and-cuda] package variant is installable.
# Checks TFCI_BAZEL_COMMON_ARGS for "gpu" or "cuda", implying that the test is
# relevant. All of the GPU test machines have CUDA installed via other means,
# so I am not sure how to verify that the dependencies themselves are valid for
# the moment.
if [[ "$TFCI_BAZEL_COMMON_ARGS" =~ gpu|cuda ]]; then
  echo "Checking to make sure tensorflow[and-cuda] is installable..."
  "$python" -m pip install "$(echo *.whl)[and-cuda]" $TFCI_PYTHON_VERIFY_PIP_INSTALL_ARGS
fi
