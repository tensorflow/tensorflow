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

#include <atomic>

#include "absl/strings/str_split.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

namespace tensorflow {

// Counter is used as a prefix for filenames.
static std::atomic<int> log_counter(0);

BridgeLoggerConfig::BridgeLoggerConfig(bool print_module_scope,
                                       bool print_after_only_on_change)
    : mlir::PassManager::IRPrinterConfig(print_module_scope,
                                         print_after_only_on_change) {
  const char* log_pass_patterns = getenv("MLIR_BRIDGE_LOG_PASS_PATTERNS");
  if (log_pass_patterns) {
    log_pass_patterns_ =
        absl::StrSplit(log_pass_patterns, ',', absl::SkipWhitespace());
  }
}

// Logs op to file with name of format
// `<log_counter>_mlir_bridge_<pass_name>_<file_suffix>.mlir`.
inline static void Log(BridgeLoggerConfig::PrintCallbackFn print_callback,
                       mlir::Pass* pass, mlir::Operation* op,
                       llvm::StringRef file_suffix) {
  std::string pass_name = pass->getName().str();

  // Add 4-digit counter as prefix so the order of the passes is obvious.
  std::string name = llvm::formatv("{0,0+4}_mlir_bridge_{1}_{2}", log_counter++,
                                   pass_name, file_suffix);

  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  if (CreateFileForDumping(name, &os, &filepath).ok()) {
    print_callback(*os);
    LOG(INFO) << "Dumped MLIR module to " << filepath;
  }
}

void BridgeLoggerConfig::printBeforeIfEnabled(mlir::Pass* pass,
                                              mlir::Operation* operation,
                                              PrintCallbackFn print_callback) {
  if (should_print(pass)) Log(print_callback, pass, operation, "before");
}

void BridgeLoggerConfig::printAfterIfEnabled(mlir::Pass* pass,
                                             mlir::Operation* operation,
                                             PrintCallbackFn print_callback) {
  if (should_print(pass)) Log(print_callback, pass, operation, "after");
}

bool BridgeLoggerConfig::should_print(mlir::Pass* pass) {
  if (log_pass_patterns_.empty()) return true;

  std::string pass_name = pass->getName().str();
  for (const auto& pattern : log_pass_patterns_) {
    if (pass_name.find(pattern) != std::string::npos) {
      // pattern matches pass
      return true;
    }
  }
  // no pattern matches pass
  VLOG(1) << "Not logging pass " << pass_name
          << " because it does not match any pattern in "
             "MLIR_BRIDGE_LOG_PASS_PATTERNS";
  return false;
}

void BridgeTimingConfig::printTiming(PrintCallbackFn printCallback) {
  std::string name = "mlir_bridge_pass_timing.txt";
  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  if (CreateFileForDumping(name, &os, &filepath).ok()) printCallback(*os);
}

}  // namespace tensorflow
