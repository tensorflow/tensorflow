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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_TF_DIALECT_TO_EXECUTOR_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_TF_DIALECT_TO_EXECUTOR_H_

#include "absl/base/attributes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

// Given the input Module op that's in the Tensorflow Dialect, convert the MLIR
// module in place to the Tensorflow Executor Dialect. Returns an OK Status if
// success, otherwise failure with an error message.
// The Tensorflow Executor Dialect is required to export an MLIR module to a
// Tensorflow GraphDef. This API will add control dependencies and verify that
// the conversion was successful. This version adds extra control dependencies
// for replication and parallel execution ops, which may slow performance.
// Prefer to use the v2 of this API.
//
// This also converts the Tensorflow Dialect MLIR into the Tensorflow Executor
// dialect that is suitable to be exported to GraphDef. Graph -> MLIR -> Graph
// is not perfectly round trippable, so this API will attempt to make the module
// exportable and verify some properties of the Tensorflow Executor MLIR that
// are required by Graph Export. It will return an error if it cannot.
//
// Input: A MLIR Module in the Tensorflow Dialect with no
// `tf_device.cluster_func` ops.
// Output: A MLIR module in the Tensorflow Executor Dialect.

ABSL_DEPRECATED(
    "Use v2/tf_dialect_to_executor.h::ExportFromTensorflowDialectToExecutor "
    "instead.")
tensorflow::Status ExportFromTensorflowDialectToExecutor(
    mlir::ModuleOp module, llvm::StringRef module_name = llvm::StringRef());

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_TF_DIALECT_TO_EXECUTOR_H_
