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
# ./define_inst.sh <filename> <baseclass> (<opname>)*

# <filename> is required, which is the file name of MLIR SPIR-V op definitions
# spec.
# <baseclass> is required. It will be the direct base class the newly defined
# op will drive from.
# If <opname> is missing, this script updates existing ones in <filename>.

# For example:
# ./define_inst.sh SPIRVArithmeticOps.td ArithmeticBianryOp OpIAdd
# ./define_inst.sh SPIRVLogicalOps.td LogicalOp OpFOrdEqual
set -e

file_name=$1
baseclass=$2

case $baseclass in
  Op | ArithmeticBinaryOp | ArithmeticUnaryOp | LogicalBinaryOp | LogicalUnaryOp | CastOp | ControlFlowOp | StructureOp | AtomicUpdateOp | AtomicUpdateWithValueOp)
  ;;
  *)
    echo "Usage : " $0 "<filename> <baseclass> (<opname>)*"
    echo "<filename> is the file name of MLIR SPIR-V op definitions spec"
    echo "<baseclass> must be one of " \
      "(Op|ArithmeticBinaryOp|ArithmeticUnaryOp|LogicalBinaryOp|LogicalUnaryOp|CastOp|ControlFlowOp|StructureOp|AtomicUpdateOp)"
    exit 1;
  ;;
esac

shift
shift

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

python3 ${current_dir}/gen_spirv_dialect.py \
  --op-td-path \
  ${current_dir}/../../include/mlir/Dialect/SPIRV/${file_name} \
  --inst-category $baseclass --new-inst "$@"

${current_dir}/define_opcodes.sh "$@"

