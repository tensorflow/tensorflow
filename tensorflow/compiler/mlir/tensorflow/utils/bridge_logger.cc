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
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

namespace tensorflow {

BridgeLoggerConfig::BridgeLoggerConfig(bool print_module_scope,
                                       bool print_after_only_on_change)
    : mlir::PassManager::IRPrinterConfig(print_module_scope,
                                         print_after_only_on_change) {}

// Logs op to file with name of format `mlir_bridge-pass_name-file_suffix.mlir`.
inline static void Log(BridgeLoggerConfig::PrintCallbackFn print_callback,
                       mlir::Pass* pass, mlir::Operation* op,
                       llvm::StringRef file_suffix) {
  std::string name =
      llvm::formatv("mlir_bridge_{0}_{1}", pass->getName(), file_suffix).str();

  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  if (CreateFileForDumping(name, &os, &filepath).ok()) print_callback(*os);
}

void BridgeLoggerConfig::printBeforeIfEnabled(mlir::Pass* pass,
                                              mlir::Operation* operation,
                                              PrintCallbackFn print_callback) {
  Log(print_callback, pass, operation, "before");
}

void BridgeLoggerConfig::printAfterIfEnabled(mlir::Pass* pass,
                                             mlir::Operation* operation,
                                             PrintCallbackFn print_callback) {
  Log(print_callback, pass, operation, "after");
}

void BridgeTimingConfig::printTiming(PrintCallbackFn printCallback) {
  std::string name = "mlir_bridge_pass_timing.txt";
  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  if (CreateFileForDumping(name, &os, &filepath).ok()) printCallback(*os);
}

}  // namespace tensorflow
