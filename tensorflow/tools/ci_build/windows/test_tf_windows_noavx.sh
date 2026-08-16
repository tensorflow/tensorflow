#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
set -ex

echo "=========================================================="
echo " Testing TensorFlow Windows CPU No-AVX Wheel in Python"
echo "=========================================================="

PYTHON_BIN=$(which python.exe 2>/dev/null || which python 2>/dev/null || echo "python")

# Find built wheel
WHEEL_FILE=$(find ./build_output /tmp/tf_wheel -name "*.whl" 2>/dev/null | head -n 1)

if [ -z "$WHEEL_FILE" ]; then
  echo "ERROR: No wheel file found to test!"
  exit 1
fi

echo "Installing wheel: $WHEEL_FILE"
$PYTHON_BIN -m pip install --force-reinstall "$WHEEL_FILE"

echo "=== Running Python Verification ==="
$PYTHON_BIN -c '
import sys
import tensorflow as tf

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"TensorFlow Git Version: {tf.__git_version__}")

# Basic Tensor creation & Math
a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
b = tf.constant([[1.0, 1.0], [0.0, 1.0]], dtype=tf.float32)
c = tf.matmul(a, b)
print(f"Matmul Result:\n{c}")

expected = tf.constant([[1.0, 3.0], [3.0, 7.0]], dtype=tf.float32)
tf.debugging.assert_near(c, expected)

# Reductions
sum_val = tf.reduce_sum(c)
print(f"ReduceSum: {sum_val.numpy()}")
assert float(sum_val.numpy()) == 14.0

print("SUCCESS: Windows TensorFlow CPU No-AVX wheel passed all runtime verification tests!")
'
