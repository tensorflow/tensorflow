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

#include <algorithm>
#include <atomic>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

constexpr char kPassFilterEnvVar[] = "MLIR_BRIDGE_LOG_PASS_FILTER";
constexpr char kStringFilterEnvVar[] = "MLIR_BRIDGE_LOG_STRING_FILTER";
constexpr char kEnableOnlyTopLevelPassesEnvVar[] =
    "MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES";

// Counter is used as a prefix for filenames.
static std::atomic<int> log_counter(0);

BridgeLoggerConfig::BridgeLoggerConfig(bool print_module_scope,
                                       bool print_after_only_on_change,
                                       mlir::OpPrintingFlags op_printing_flags)
    : mlir::PassManager::IRPrinterConfig(
          print_module_scope, print_after_only_on_change,
          /*printAfterOnlyOnFailure=*/false, op_printing_flags),
      pass_filter_(GetFilter(kPassFilterEnvVar)),
      string_filter_(GetFilter(kStringFilterEnvVar)) {
  setenv(/*name=*/kEnableOnlyTopLevelPassesEnvVar,
         /*value=*/"false", /*overwrite=*/0);
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
                                              mlir::Operation* op,
                                              PrintCallbackFn print_callback) {
  if (ShouldPrint(pass, op)) Log(print_callback, pass, op, "before");
}

void BridgeLoggerConfig::printAfterIfEnabled(mlir::Pass* pass,
                                             mlir::Operation* op,
                                             PrintCallbackFn print_callback) {
  if (ShouldPrint(pass, op)) Log(print_callback, pass, op, "after");
}

std::vector<std::string> BridgeLoggerConfig::GetFilter(
    const std::string& env_var) {
  std::vector<std::string> filter;
  const char* filter_str = getenv(env_var.c_str());
  if (filter_str) {
    filter = absl::StrSplit(filter_str, ';', absl::SkipWhitespace());
  }
  return filter;
}

bool BridgeLoggerConfig::ShouldOnlyDumpTopLevelPasses() {
  const char* env_var = getenv(kEnableOnlyTopLevelPassesEnvVar);
  std::string value(env_var);
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  // Return true if value is "1" or "true"; otherwise, false.
  return value == "1" || value == "true";
}

bool BridgeLoggerConfig::MatchesFilter(const std::string& str,
                                       const std::vector<std::string>& filter,
                                       bool exact_match) {
  if (filter.empty()) return true;
  for (const std::string& filter_str : filter) {
    if (str == filter_str) return true;
    if (!exact_match && str.find(filter_str) != std::string::npos) return true;
  }
  return false;
}

bool BridgeLoggerConfig::ShouldPrint(mlir::Pass* pass, mlir::Operation* op) {
  // Check pass filter first since it's cheaper.
  std::string pass_name = pass->getName().str();
  if (!MatchesFilter(pass_name, pass_filter_, /*exact_match=*/true)) {
    // No string in filter matches pass name.
    VLOG(1) << "Not logging invocation of pass `" << pass_name
            << "` because the pass name does not match any string in "
               "`MLIR_BRIDGE_LOG_PASS_FILTER`";
    return false;
  }
  if (ShouldOnlyDumpTopLevelPasses()) {
    // Check if the operation is the top-level module.
    // Top-level module has no parent.
    if (op->getParentOp() != nullptr) {
      // This is a nested operation; do not print.
      VLOG(1) << "Not logging invocation of pass `" << pass_name
              << "` because it is applied to a nested operation, due to "
                 "`MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES` being set "
                 "to true";
      return false;
    }
  }
  if (!string_filter_.empty()) {
    std::string serialized_op;
    llvm::raw_string_ostream os(serialized_op);
    op->print(os);
    if (!MatchesFilter(serialized_op, string_filter_, /*exact_match=*/false)) {
      // No string in filter was found in serialized `op`.
      VLOG(1) << "Not logging invocation of pass `" << pass_name
              << "` because the serialized operation on which the pass is "
                 "invoked does not contain any of the strings specified by "
                 "MLIR_BRIDGE_LOG_STRING_FILTER";
      return false;
    }
  }
  return true;
}

}  // namespace tensorflow
