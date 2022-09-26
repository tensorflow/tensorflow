/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements utilities for canonicalization of ScatterOp.

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MHLO_SCATTER_UTILS_H_
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MHLO_SCATTER_UTILS_H_

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace mhlo {

// Checks if the scatter has the following characteristics:
// - scatter_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - inserted_window_dims is []
// - update_window_dims is [0, 1, ...]
// - scatter_dims_to_operand_dims is [0, 1, ...]
bool isCanonicalScatter(ScatterOp scatterOp);

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MHLO_SCATTER_UTILS_H_
