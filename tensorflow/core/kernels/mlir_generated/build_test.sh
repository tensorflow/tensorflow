#!/bin/bash
#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# A script to catch kernel_gen failures as a test rather than build failure.
set -e

# KernelGen binary
TF_TO_KERNEL="$1"
OUTPUT_FILE="${TEST_TMPDIR}/output.mlir"
INPUT="$2"
PLATFORM="$3"

# Do something
${TF_TO_KERNEL} --input=${INPUT} --output=${OUTPUT_FILE} ${PLATFORM} "${@:4}"  || die "Failed to generate kernel"

# Check something
[ -s ${OUTPUT_FILE} ] || die "output file was empty"

echo "PASS"
