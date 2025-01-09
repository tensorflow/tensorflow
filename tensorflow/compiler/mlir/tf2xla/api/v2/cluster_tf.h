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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_CLUSTER_TF_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_CLUSTER_TF_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v2/device_type.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

// Run all the passes involved in transforming the graph before execution so
// that it is suitable for targeting devices when called with the TF 2 Function
// API. Users that need clustering with the Session API should use the v1 Bridge
// API. These transformations take as input a Tensorflow Graph as an MLIR Module
// and transforms the module in place to cluster the given ops for compilation
// that is compatible with the given device_type. The MLIR should be in the TF
// Executor Dialect for graph nodes and edges or be in TF Functional already.
// Individual Op inside a node should be the Tensorflow Functional Dialect. The
// output MLIR is in the TF Functional Dialect. Returns OkStatus if passed,
// otherwise an error.
//
// Inputs:
//   module - The MLIR Module that will be clustered. Expected to be in TF
//   Executor Dialect or TF Functional Dialect. Will convert to TF Functional.
//   is_supported_by_replicated_brige - If the graph targets the replicated
//   bridge. Set it to true for replicated/partitioned graphs. e.g. replicated
//   and single-core TPU graphs. Set this to false if the graph is not
//   replicated, e.g. CPU/GPU graphs. is_in_fallback_enabled_mode - Whether this
//   was called with fallback to the non-MLIR Bridge. This is just for logging
//   purposes and doesn't affect logic. module_name - What the input module name
//   is for debugging help.
//
// Output: Modifies the input module in place with clustered operations.
//   status - Whether the transformation to cluster the input MLIR module was
//   successful.
absl::Status RunFunctionTf2xlaClusteringBridge(
    mlir::ModuleOp module, bool is_supported_by_replicated_brige,
    bool is_in_fallback_enabled_mode,
    llvm::StringRef module_name = llvm::StringRef());
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_CLUSTER_TF_H_
