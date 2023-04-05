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

#ifndef TENSORFLOW_DTENSOR_MLIR_OP_UTILS_H_
#define TENSORFLOW_DTENSOR_MLIR_OP_UTILS_H_

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

// Computes a deterministic hash of the given operation.
uint64_t OpHash(mlir::Operation* op);

inline std::string OpName(mlir::Operation* op) {
  auto ref = op->getName().getStringRef();
  return ref.str();
}

// Returns FuncOp if `op` is a callable.
absl::optional<mlir::func::FuncOp> MaybeFindFunction(mlir::Operation* op);

// DTensorLayout only conveys layout information of tensors which is no
// longer needed after SPMD expansion. As so, remove all layouts from
// graph.
//
// If `remove_xla_spmd_layouts` is set to false, then Layouts that have
// Mesh config with use_xla_spmd=True will not be deleted.
//
// Removes tf.DTensorLayout op and forwards it's input to it's users.
// For example:
//   %0 = tf.A()
//   %1 = tf.DTensorLayout(%0)
//   %2 = tf.B(%1)
//
// Will be converted to:
//   %0 = tf.A()
//   %2 = tf.B(%0)
void RemoveDTensorLayoutOps(mlir::ModuleOp module,
                            bool remove_xla_spmd_layouts);

// Canonicalizer and DCE transformation passes may removed ops in the graph and
// result in multiple consecutive DTensorLayout ops. Detect all such cases and
// replace unnecessary DTensorLayout ops with Identity ops.
//
// Removes tf.DTensorLayouts and inserts a tf.Identity.
// For example:
//   %0 = tf.DTensorLayout(arg0)
//   %1 = tf.DTensorLayout(%0)
//   %2 = tf.Add(%1, %1)
//
// Will be converted to:
//   %0 = tf.Identity(arg0)
//   %1 = tf.DTensorLayout(%0)
//   %2 = tf.Add(%1, %1)
mlir::LogicalResult ReplaceAuxiliaryDTensorLayoutOpsWithIdentity(
    mlir::ModuleOp module);
}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_OP_UTILS_H_
