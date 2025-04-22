/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/instrumentations/save_report.h"

#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/report.h"

namespace mlir::quant::stablehlo {
namespace {

// Converts `std::optional<absl::string_view>` to `std::optional<std::string>`.
// A `std::nullopt` is returned when `view` is `std::nullopt`.
std::optional<std::string> OptionalStringViewToOptionalString(
    std::optional<absl::string_view> view) {
  if (view == std::nullopt) return std::nullopt;
  return std::make_optional<std::string>(*view);
}

// Whether the pass is `QuantizeCompositeFunctionPass`.
bool IsQuantizeCompositeFunctionPass(Pass* /*absl_nullable*/ pass,
                                     Operation* /*absl_nullable*/ op) {
  // It is known that `op` is `ModuleOp` when `pass` is
  // `QuantizeCompositeFunctionPass`, but the check is still performed to be
  // defensive.
  return pass != nullptr &&
         pass->getArgument() == "stablehlo-quantize-composite-functions" &&
         isa_and_nonnull<ModuleOp>(op);
}

// Report is saved only when:
// * After running `QuantizeCompositeFunctionPass`.
// * The pass is run on `ModuleOp`.
// * `file_path` is not `nullopt`.
bool ShouldSaveReport(Pass* /*absl_nullable*/ pass, Operation* /*absl_nullable*/ op,
                      const std::optional<std::string>& file_path) {
  return file_path != std::nullopt && IsQuantizeCompositeFunctionPass(pass, op);
}

void SaveReport(const QuantizationReport& report,
                const absl::string_view file_path) {
  if (const absl::Status save_status = report.Save(file_path);
      save_status.ok()) {
    LOG(INFO) << "Successfully saved quantization report to: " << file_path;
  } else {
    LOG(ERROR) << "Failed to save quantization report to: " << file_path
               << " with status: " << save_status;
  }
}

}  // namespace

SaveQuantizationReportInstrumentation::SaveQuantizationReportInstrumentation(
    std::optional<absl::string_view> file_path)
    : file_path_(OptionalStringViewToOptionalString(file_path)) {}

void SaveQuantizationReportInstrumentation::runAfterPass(Pass* pass,
                                                         Operation* op) {
  // Only run after `QuantizeCompositeFunctionPass`.
  if (!IsQuantizeCompositeFunctionPass(pass, op)) return;

  auto module_op = cast<ModuleOp>(op);
  const QuantizationReport report(module_op);

  // Print a human-readable report to stdout regardless of whether the report
  // is saved to file.
  report.Print();

  // Exit early if the report should not be saved to file.
  if (!ShouldSaveReport(pass, op, file_path_)) return;

  SaveReport(report, *file_path_);
}

}  // namespace mlir::quant::stablehlo
