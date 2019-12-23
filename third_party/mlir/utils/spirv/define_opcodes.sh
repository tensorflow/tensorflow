#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
