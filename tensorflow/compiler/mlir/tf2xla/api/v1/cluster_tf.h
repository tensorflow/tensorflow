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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_CLUSTER_TF_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_CLUSTER_TF_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

// Run all the passes involved in transforming the graph before execution so
// that it is suitable for targeting devices when called via the TF1 Session
// API.
// These transformations take as input a Tensorflow Graph as an MLIR Module
// and transforms the module in place to cluster the given ops for compilation
// that is compatible with the given device_type. The MLIR should be in the TF
// Executor Dialect for graph nodes and edges or TF Functional. It will convert
// to TF Functional internally. Individual Op inside a node should be the
// Tensorflow Dialect. The output MLIR is in the TF Functional Dialect.  The
// input MLIR should not have infeed and outfeed ops, which are unsupported via
// this API. Returns OkStatus if passed, otherwise an error.
tensorflow::Status RunSessionTf2xlaClusteringBridge(
    mlir::ModuleOp module, bool is_in_fallback_enabled_mode);

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_CLUSTER_TF_H_
