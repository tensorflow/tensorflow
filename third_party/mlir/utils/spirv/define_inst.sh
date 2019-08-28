#!/bin/bash

# Copyright 2019 The MLIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for defining a new op using SPIR-V spec from the Internet.
#
# Run as:
# ./define_inst.sh <opname>

# For example:
# ./define_inst.sh OpIAdd
#
# If <opname> is missing, this script updates existing ones.

set -e

new_op=$1

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

python3 ${current_dir}/gen_spirv_dialect.py \
  --op-td-path ${current_dir}/../../include/mlir/Dialect/SPIRV/SPIRVOps.td \
  --new-inst "${new_op}"
