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

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/lib/core/status.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

using mlir::OpPassManager;
using mlir::TF::StandardPipelineOptions;

void CreateLowerClusterToRuntimeOpsPassPipeline(
    OpPassManager &pm, const StandardPipelineOptions &options) {}

}  // namespace

tensorflow::Status RunLowerClusterToRuntimeOpsPassPipeline(
    mlir::ModuleOp module, llvm::StringRef module_name) {
  return tsl::OkStatus();
}

void RegisterLowerClusterToRuntimeOpsPassPipeline() {
  static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
      "tfrt-lower-cluster-to-runtime-ops",
      "Run all the passes involved after the clustering transformations from "
      "the TF2XLA Bridge. Takes as input a Module with tf_device.cluster ops "
      "and outputs TFRT runtime ops such as TPUCompile",
      CreateLowerClusterToRuntimeOpsPassPipeline);
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
