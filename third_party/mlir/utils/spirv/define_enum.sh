#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script for defining a new enum attr using SPIR-V spec from the Internet.
#
# Run as:
# ./define_enum.sh <enum-class-name>
#
# The 'operand_kinds' dict of spirv.core.grammar.json contains all supported
# SPIR-V enum classes.
#
# If <enum-name> is missing, this script updates existing ones.

set -e

new_enum=$1

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

python3 ${current_dir}/gen_spirv_dialect.py \
  --base-td-path ${current_dir}/../../include/mlir/Dialect/SPIRV/SPIRVBase.td \
  --new-enum "${new_enum}"
