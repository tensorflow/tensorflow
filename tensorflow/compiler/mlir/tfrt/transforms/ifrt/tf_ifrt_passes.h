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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF_IFRT_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF_IFRT_PASSES_H_

#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace tensorflow {
namespace ifrt_serving {

// Register all passes.
void RegisterTfIfrtPasses();

// Convert tf_device.cluster_func to tf.ifrt_program_call.
// The callee function is converted to a ifrt_program.
absl::Status RunClusterToIfrtRuntimeOpsPassPipeline(
    mlir::ModuleOp module, llvm::StringRef module_name = llvm::StringRef());

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF_IFRT_PASSES_H_
