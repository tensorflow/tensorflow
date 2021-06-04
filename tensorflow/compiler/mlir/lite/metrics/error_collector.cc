/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/metrics/error_collector.h"

#include <vector>

#include "absl/strings/str_split.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

// The signature contains namespaces (Ex: mlir::TFL::(anonymous namespace)::).
// So only extract the function name as the pass name.
inline std::string extract_pass_name(const std::string &signature) {
  const std::vector<std::string> &v = absl::StrSplit(signature, "::");
  return v.back();
}
}  // namespace

ErrorCollectorInstrumentation::ErrorCollectorInstrumentation(
    MLIRContext *context) {
  handler_.reset(new ScopedDiagnosticHandler(context, [this](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      Location loc = diag.getLocation();
      std::string op_name, error_code;
      if (loc_to_name_.count(loc)) {
        op_name = loc_to_name_[loc];
      }

      for (auto &note : diag.getNotes()) {
        if (note.str().rfind(kErrorCodePrefix, 0) == 0) {
          error_code = note.str().substr(sizeof(kErrorCodePrefix) - 1);
        }
      }

      ErrorCode error_code_enum = ConverterErrorData::UNKNOWN;
      bool has_valid_error_code =
          ConverterErrorData::ErrorCode_Parse(error_code, &error_code_enum);
      if (!op_name.empty() || has_valid_error_code) {
        AppendNewError(/*pass_name=*/"", diag.str(), error_code_enum, op_name);
      } else {
        common_error_message_ += diag.str();
        common_error_message_ += "\n";
      }
    }
    return failure();
  }));
}

void ErrorCollectorInstrumentation::runBeforePass(Pass *pass,
                                                  Operation *module) {
  // Find the op names with tf or tfl dialect prefix, Ex: "tf.Abs" or "tfl.Abs".
  auto collectOps = [this](Operation *op) {
    const auto &op_name = op->getName().getStringRef().str();
    if (absl::StartsWith(op_name, "tf.") || absl::StartsWith(op_name, "tfl.")) {
      loc_to_name_.emplace(op->getLoc(), op_name);
    }
  };

  for (auto &region : module->getRegions()) {
    region.walk(collectOps);
  }
}

void ErrorCollectorInstrumentation::runAfterPass(Pass *pass,
                                                 Operation *module) {
  loc_to_name_.clear();
  errors_.clear();
  common_error_message_.clear();
}

void ErrorCollectorInstrumentation::runAfterPassFailed(Pass *pass,
                                                       Operation *module) {
  std::string pass_name = extract_pass_name(pass->getName().str());
  for (auto &error : errors_) {
    error.set_subcomponent(pass_name);
  }

  // Create a new error if no errors collected yet.
  if (errors_.empty() && !common_error_message_.empty()) {
    AppendNewError(pass_name, common_error_message_,
                   ConverterErrorData::UNKNOWN, /*op_name=*/"");
  }

  // Report to ErrorCollector.
  ErrorCollector *collector = GetErrorCollector();
  for (auto &error : errors_) {
    collector->ReportError(error);
  }
  loc_to_name_.clear();
  errors_.clear();
  common_error_message_.clear();
}

ErrorCollector *GetErrorCollector() {
  static ErrorCollector *global_error_collector = new ErrorCollector();
  return global_error_collector;
}
}  // namespace TFL
}  // namespace mlir
