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

#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

namespace tensorflow {

// Logs op to file with name of format `mlir_bridge-pass_name-file_suffix.mlir`.
inline static void Log(mlir::Pass* pass, mlir::Operation* op,
                       llvm::StringRef file_suffix) {
  DumpMlirOpToFile(
      llvm::formatv("mlir_bridge-{0}-{1}", pass->getName(), file_suffix).str(),
      op);
}

void BridgeLogger::runBeforePass(mlir::Pass* pass, mlir::Operation* op) {
  Log(pass, op, "before");
}

void BridgeLogger::runAfterPass(mlir::Pass* pass, mlir::Operation* op) {
  Log(pass, op, "after");
}

}  // namespace tensorflow
