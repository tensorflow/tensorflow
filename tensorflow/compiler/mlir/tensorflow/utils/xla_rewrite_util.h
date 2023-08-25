/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_REWRITE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_REWRITE_UTIL_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
// Erase rewritten ClusterFuncOp(s). If TPUPartitionedInputV2Op /
// TPUPartitionedOutputV2Op are present, they must be removed along with the
// ClusterFuncOp(s).
mlir::LogicalResult EraseClusterFuncs(
    llvm::MutableArrayRef<mlir::tf_device::ClusterFuncOp> to_be_erased);

// Move child processes of the ParallelExecute that do not change. These are all
// children except for the child with the ClusterFunc.
// Returns the index of the child with the ClusterFunc.
int MovePreservedParallelExecuteChildren(
    int num_cores_per_replica,
    llvm::SmallVector<mlir::Type, 8>& concatenated_output_types,
    mlir::OpBuilder* builder, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute,
    mlir::tf_device::ParallelExecuteOp* new_parallel_execute);

// Wraps single op in `tf_device.launch` for explicit device assignment.
mlir::tf_device::LaunchOp WrapOpInLaunch(mlir::OpBuilder* builder,
                                         mlir::Location loc,
                                         mlir::Operation* op,
                                         llvm::StringRef device);

}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_REWRITE_UTIL_H_
