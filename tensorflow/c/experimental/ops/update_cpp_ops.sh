#!/bin/bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Script to manually generate the C++ ops.

set -e

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"
api_dir="${current_dir}/../../../core/api_def/base_api"

generate="bazel run \
  //tensorflow/c/experimental/ops/gen:generate_cpp -- \
  --output_dir="${current_dir}" \
  --api_dirs="${api_dir}" \
  --source_dir=third_party/tensorflow"

${generate} \
  --category=array \
  Identity \
  IdentityN \
  ZerosLike \
  Shape \
  ExpandDims \
  OnesLike

${generate} \
  --category=math \
  Mul \
  Conj \
  AddV2 \
  MatMul \
  Neg \
  Sum \
  Sub \
  Div \
  DivNoNan \
  Exp \
  Sqrt \
  SqrtGrad \
  Log1p

${generate} \
  --category=nn \
  SparseSoftmaxCrossEntropyWithLogits \
  ReluGrad \
  Relu \
  BiasAdd \
  BiasAddGrad

${generate} \
  --category=resource_variable \
  VarHandleOp \
  ReadVariableOp \
  AssignVariableOp \
  DestroyResourceOp
