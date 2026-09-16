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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_PASS_DEBUG_INSTRUMENTATION_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_PASS_DEBUG_INSTRUMENTATION_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/utils/error_util.h"

namespace mlir::TFL {

class PassDebugInstrumentation : public mlir::PassInstrumentation {
 public:
  PassDebugInstrumentation(std::string* name, std::string* arg,
                           std::string* failing_func,
                           mlir::OwningOpRef<mlir::ModuleOp>* clean_module,
                           bool enable_debug, absl::string_view debug_dir,
                           absl::string_view print_before,
                           absl::string_view print_after,
                           int64_t elide_elements_larger_than = 8,
                           int64_t elide_resource_strings_larger_than = 64);

  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override;
  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override;
  void runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) override;

 private:
  void DumpIrToFile(mlir::Pass* pass, mlir::Operation* op,
                    llvm::StringRef suffix);
  bool MatchPassRegex(llvm::StringRef pass_name, const llvm::Regex* regex);

  std::string* name_;
  std::string* arg_;
  std::string* failing_func_;
  mlir::OwningOpRef<mlir::ModuleOp>* clean_module_;
  bool enable_debug_;
  std::string debug_dir_;
  std::unique_ptr<llvm::Regex> print_before_regex_;
  std::unique_ptr<llvm::Regex> print_after_regex_;
  int64_t elide_elements_larger_than_ = 8;
  int64_t elide_resource_strings_larger_than_ = 64;
  int step_counter_ = 0;
};

class SerializationDiagHandler : public mlir::BaseScopedDiagnosticHandler {
 public:
  explicit SerializationDiagHandler(mlir::MLIRContext* ctx, std::string* out)
      : BaseScopedDiagnosticHandler(ctx), out_(out) {
    setHandler([this](mlir::Diagnostic& diag) {
      if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
        if (!out_->empty()) *out_ += '\n';
        *out_ += diag.str();
      }
      return mlir::failure();
    });
  }

 private:
  std::string* out_;
};

// Coordinating class that encapsulates pass failure diagnostic variables and
// coordinates with PassDebugInstrumentation and ConversionFailureReporter.
class PipelineFailureCoordinator {
 public:
  PipelineFailureCoordinator(const std::string& debug_dir, bool enable_debug,
                             int64_t elide_elements_larger_than = 8,
                             int64_t elide_resource_strings_larger_than = 64)
      : debug_dir_(debug_dir),
        enable_debug_(enable_debug),
        elide_elements_larger_than_(elide_elements_larger_than),
        elide_resource_strings_larger_than_(
            elide_resource_strings_larger_than) {}

  std::unique_ptr<mlir::PassInstrumentation> CreateInstrumentation(
      absl::string_view print_before_pattern,
      absl::string_view print_after_pattern);

  absl::Status ReportFailure(mlir::ModuleOp fallback_module,
                             const absl::Status& pass_status) const;

  absl::Status ReportSerializationFailure(mlir::ModuleOp module,
                                          const absl::Status& status,
                                          absl::string_view diag_errors) const;

 private:
  std::string debug_dir_;
  bool enable_debug_;
  int64_t elide_elements_larger_than_ = 8;
  int64_t elide_resource_strings_larger_than_ = 64;
  std::string failing_pass_name_;
  std::string failing_pass_arg_;
  std::string failing_function_name_;
  mlir::OwningOpRef<mlir::ModuleOp> pre_pass_clean_module_;
};

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_PASS_DEBUG_INSTRUMENTATION_H_
