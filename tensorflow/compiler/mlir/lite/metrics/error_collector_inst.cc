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
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"

#include <string>
#include <vector>

#include "absl/strings/match.h"
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

// Errors raised by emitOpError start with "'<dialect>.<op>' op". Returns an
// empty string if the pattern is not found or the operator is not in tf or tfl
// dialect.
inline std::string extract_op_name_from_error_message(
    const std::string &error_message) {
  int end_pos = error_message.find("' op");
  if ((absl::StartsWith(error_message, "'tf.") ||
       absl::StartsWith(error_message, "'tfl.")) &&
      end_pos != std::string::npos) {
    return error_message.substr(1, end_pos - 1);
  }
  return "";
}

// Only notes with character count smaller than kMaxAcceptedNoteSize will be
// appended to the error message.
const int kMaxAcceptedNoteSize = 1024;
}  // namespace

ErrorCollectorInstrumentation::ErrorCollectorInstrumentation(
    MLIRContext *context)
    : error_collector_(ErrorCollector::GetErrorCollector()) {
  handler_.reset(new ScopedDiagnosticHandler(context, [this](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      Location loc = diag.getLocation();
      std::string error_message = diag.str();
      std::string op_name, error_code;
      if (loc_to_name_.count(loc)) {
        op_name = loc_to_name_[loc];
      } else {
        op_name = extract_op_name_from_error_message(diag.str());
      }

      for (const auto &note : diag.getNotes()) {
        const std::string note_str = note.str();
        if (note_str.rfind(kErrorCodePrefix, 0) == 0) {
          error_code = note_str.substr(sizeof(kErrorCodePrefix) - 1);
        }

        error_message += "\n";
        if (note_str.size() <= kMaxAcceptedNoteSize) {
          error_message += note_str;
        } else {
          error_message += note_str.substr(0, kMaxAcceptedNoteSize);
          error_message += "...";
        }
      }

      ErrorCode error_code_enum = ConverterErrorData::UNKNOWN;
      bool has_valid_error_code =
          ConverterErrorData::ErrorCode_Parse(error_code, &error_code_enum);
      if (!op_name.empty() || has_valid_error_code) {
        error_collector_->ReportError(NewConverterErrorData(
            pass_name_, error_message, error_code_enum, op_name, loc));
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

  pass_name_ = extract_pass_name(pass->getName().str());
  error_collector_->Clear();
}

void ErrorCollectorInstrumentation::runAfterPass(Pass *pass,
                                                 Operation *module) {
  loc_to_name_.clear();
  pass_name_.clear();
  common_error_message_.clear();
  error_collector_->Clear();
}

void ErrorCollectorInstrumentation::runAfterPassFailed(Pass *pass,
                                                       Operation *module) {
  // Create a new error if no errors collected yet.
  if (error_collector_->CollectedErrors().empty() &&
      !common_error_message_.empty()) {
    error_collector_->ReportError(NewConverterErrorData(
        pass_name_, common_error_message_, ConverterErrorData::UNKNOWN,
        /*op_name=*/"", module->getLoc()));
  }

  loc_to_name_.clear();
  pass_name_.clear();
  common_error_message_.clear();
}

}  // namespace TFL
}  // namespace mlir
