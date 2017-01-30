#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Bash unit tests for TensorFlow Debugger (tfdbg) Python examples that do not
# involve downloading data.

set -e


DEBUG_FIBONACCI_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_fibonacci"

# Override the default ui_type=curses to allow the test to pass in a tty-less
# test environment.
cat << EOF | "${DEBUG_FIBONACCI_BIN}" --ui_type=readline
run
exit
EOF


DEBUG_ERRORS_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_errors"

cat << EOF | "${DEBUG_ERRORS_BIN}" --error=no_error --ui_type=readline
run
exit
EOF


DEBUG_MNIST_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_mnist"

# Use a large enough "run -t" number to let the process end properly.
cat << EOF | "${DEBUG_MNIST_BIN}" --debug --fake_data --ui_type=readline
run -f has_inf_or_nan
run -t 1000
EOF


DEBUG_TFLEARN_IRIS_BIN="$TEST_SRCDIR/org_tensorflow/tensorflow/python/debug/debug_tflearn_iris"

cat << EOF | "${DEBUG_TFLEARN_IRIS_BIN}" --debug --fake_data --train_steps=2 --ui_type=readline
run -f has_inf_or_nan
EOF
