/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_SPLIT_INTO_ISLAND_PER_OP_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_SPLIT_INTO_ISLAND_PER_OP_PASS_H_

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {
namespace TF {

// Converts a single island into multiple islands (one for each op).
void SplitIsland(mlir::tf_executor::IslandOp island_op,
                 mlir::tf_executor::ControlType control_type);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_SPLIT_INTO_ISLAND_PER_OP_PASS_H_
