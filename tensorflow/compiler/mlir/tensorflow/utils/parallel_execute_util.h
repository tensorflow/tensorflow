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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_PARALLEL_EXECUTE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_PARALLEL_EXECUTE_UTIL_H_

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

namespace mlir {
namespace TFTPU {

// TODO(b/243076653): Once the ParallelExecute is added do not remove it. This
//   means BuildSingletonParallelExecuteOp will be used in one location, and
//   RemoveSingletonParallelExecuteOp can be removed.

// Wrap `cluster_func` in a `ParallelExecute` with only one child. This
// can be used to canonicalize IR, so there is always one `ParallelExecute`.
tf_device::ParallelExecuteOp BuildParallelExecuteOp(
    tf_device::ClusterFuncOp cluster_func, OpBuilder* builder);

// Unwrap `parallel_execute`'s contents if it only has one child.
LogicalResult RemoveSingletonParallelExecuteOp(
    tf_device::ParallelExecuteOp parallel_execute, OpBuilder* builder);

}  // namespace TFTPU
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_PARALLEL_EXECUTE_UTIL_H_
