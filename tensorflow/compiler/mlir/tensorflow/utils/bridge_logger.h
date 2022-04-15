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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_BRIDGE_LOGGER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_BRIDGE_LOGGER_H_

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/Timing.h"  // from @llvm-project

namespace tensorflow {

// Logger for logging MLIR modules before and after passes in MLIR TPU bridge.
//
// The IR logging can be restricted to a particular set of pass invocations via
// filters that are specified with the `MLIR_BRIDGE_LOG_PASS_FILTER` and
// `MLIR_BRIDGE_LOG_STRING_FILTER` environment variables.
// `MLIR_BRIDGE_LOG_PASS_FILTER` takes a semicolon-separated list of pass class
// names, `MLIR_BRIDGE_LOG_STRING_FILTER` takes a semicolon-separated list of
// strings, and IR is only dumped for a pass invocation if the pass name exactly
// matches any of the provided pass names and if the serialized operation on
// which the pass is invoked contains any of the specified strings as a
// substring. An empty list is interpreted as no restriction. The string filter
// can be handy e.g. if one is only interested in a certain function or when
// checking where a certain attribute gets lost. Note that we use a semicolon
// instead of comma as the separator to allow strings that contain commas (which
// frequently appear in MLIR). The strings can contain any characters (including
// spaces) except semicolons.
//
// Example: Setting the environment variables
// `MLIR_BRIDGE_LOG_PASS_FILTER="LegalizeTF;Canonicalizer"` and
// `MLIR_BRIDGE_LOG_STRING_FILTER="my_string"` will dump IR only for invocations
// of `LegalizeTF` and `Canonicalizer` where the string `my_string` is contained
// in the serialized operation on which the pass is invoked. For verbose log
// level >= 1, `bridge_logger.cc` prints details about pass invocations for
// which the IR dumping was skipped because of a filter.
class BridgeLoggerConfig : public mlir::PassManager::IRPrinterConfig {
 public:
  explicit BridgeLoggerConfig(bool print_module_scope = false,
                              bool print_after_only_on_change = true);

  // A hook that may be overridden by a derived config that checks if the IR
  // of 'operation' should be dumped *before* the pass 'pass' has been
  // executed. If the IR should be dumped, 'print_callback' should be invoked
  // with the stream to dump into.
  void printBeforeIfEnabled(mlir::Pass* pass, mlir::Operation* op,
                            PrintCallbackFn print_callback) override;

  // A hook that may be overridden by a derived config that checks if the IR
  // of 'operation' should be dumped *after* the pass 'pass' has been
  // executed. If the IR should be dumped, 'print_callback' should be invoked
  // with the stream to dump into.
  void printAfterIfEnabled(mlir::Pass* pass, mlir::Operation* op,
                           PrintCallbackFn print_callback) override;

  // Returns `true` iff we should log IR for given `pass` and `op`.
  // Note: Visibility of this function is public for use in unit testing.
  bool ShouldPrint(mlir::Pass* pass, mlir::Operation* op);

 private:
  // Get `filter` encoded by environment variable `env_var`.
  static std::vector<std::string> GetFilter(const std::string& env_var);
  // Returns `true` iff any of the strings in `filter` matches `str`, either
  // exactly or as a substring, depending on `exact_match`.
  static bool MatchesFilter(const std::string& str,
                            const std::vector<std::string>& filter,
                            bool exact_match);

  // Only log pass invocations whose pass name exactly matches any string in
  // `pass_filter_` (or when `pass_filter_` is empty).
  const std::vector<std::string> pass_filter_;
  // Only log pass invocations where the serialized operation on which the pass
  // is invoked contains any of the specified strings as a substring (or when
  // `string_filter_` is empty).
  const std::vector<std::string> string_filter_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_BRIDGE_LOGGER_H_
