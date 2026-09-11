/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERSION_FAILURE_REPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERSION_FAILURE_REPORTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir::TFL {

struct FailureReport {
  std::string stage;
  std::string status_code;
  std::string failing_pass;
  std::string failing_pass_arg;
  std::string failing_function;
  std::string raw_error;
  std::string elided_mlir_file;
  std::string bytecode_file;

  struct PrimaryError {
    std::string location;
    std::string severity;
    std::string message;
    std::string code_snippet;
  } primary_error;

  struct CallStackFrame {
    std::string location;
    std::string code_snippet;
  };
  std::vector<CallStackFrame> call_stack;

  struct OperationContext {
    std::string location;
    std::string operation;
  } operation_context;
};

class ConversionFailureReporter {
 public:
  // Parses a raw MLIR error diagnostic message into structured fields.
  static FailureReport ParseDiagnostic(absl::string_view raw_error,
                                       absl::string_view stage,
                                       absl::string_view status_code,
                                       absl::string_view failing_pass = "",
                                       absl::string_view failing_pass_arg = "",
                                       absl::string_view failing_function = "");

  // Returns working_dir if non-empty, or generates a unique directory name
  // under /tmp based on timestamp and process ID.
  static std::string GetOrCreateDebugDir(absl::string_view working_dir);

  // Writes the structured FailureReport, elided MLIR, and bytecode module.
  static void WriteFailureJson(absl::string_view working_dir,
                               mlir::ModuleOp module,
                               absl::string_view raw_error,
                               absl::string_view stage,
                               absl::string_view status_code,
                               absl::string_view failing_pass = "",
                               absl::string_view failing_pass_arg = "",
                               bool write_module_artifacts = true,
                               absl::string_view failing_function = "",
                               int64_t elide_elements_larger_than = 8,
                               int64_t elide_resource_strings_larger_than = 64);
};

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERSION_FAILURE_REPORTER_H_
