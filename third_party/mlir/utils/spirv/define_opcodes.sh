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

# Script for defining map for opname to opcode using SPIR-V spec from the
# Internet
#
# Run as:
# ./define_opcode.sh (<op-name>)*
#
# For example:
# ./define_opcode.sh OpTypeVoid OpTypeFunction
#
# If no op-name is specified, the existing opcodes are updated
#
# The 'instructions' list of spirv.core.grammar.json contains all instructions
# in SPIR-V

set -e

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

python3 ${current_dir}/gen_spirv_dialect.py \
  --base-td-path ${current_dir}/../../include/mlir/Dialect/SPIRV/SPIRVBase.td \
  --new-opcode $@
