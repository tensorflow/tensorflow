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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/report.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo {
namespace {

using ::mlir::tf_quant::GetQuantizationMethod;
using ::mlir::tf_quant::kCompositeFuncPrefix;
using ::mlir::tf_quant::kOriginalStablehloEntryFunctionAttrName;
using ::mlir::tf_quant::kQuantizedFuncPrefix;
using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::QuantizationResult;
using ::stablehlo::quantization::QuantizationResults;
using ::stablehlo::quantization::io::WriteStringToFile;
using ::tsl::protobuf::TextFormat;

// Given a `quantized_func_name` that starts with `kQuantizedFuncPrefix`,
// converts `kQuantizedFuncPrefix` to `kCompositeFuncPrefix`.
std::string GetCompositeFunctionName(const StringRef quantized_func_name) {
  return Twine(kCompositeFuncPrefix)
      .concat(quantized_func_name.rsplit(kQuantizedFuncPrefix).second)
      .str();
}

// Retrieves `QuantizationResult` from `call_op`. If the callee's name starts
// with `kQuantizedFuncPrefix` then a `QuantizationResult` will be returned with
// its `name` field set to the callee's name reverted back to the lifted
// function's name. Also, `call_op` must have the `kQuantizationMethodAttr`
// attribute, which is deserialized as `Method` and set in the returned
// `QuantizationResult`. Otherwise, it returns `std::nullopt`.
std::optional<QuantizationResult> GetQuantizationResult(func::CallOp call_op) {
  const StringRef callee_name = call_op.getCalleeAttr().getValue();
  if (!callee_name.starts_with(kQuantizedFuncPrefix)) {
    return std::nullopt;  // `call_op` is not a quantized function call.
  }

  absl::StatusOr<Method> method = GetQuantizationMethod(call_op);
  if (!method.ok()) {
    call_op->emitError() << "Failed to get quantization method: "
                         << method.status().ToString();
    return std::nullopt;
  }

  QuantizationResult result{};
  result.mutable_quantizable_unit()->set_name(
      GetCompositeFunctionName(callee_name));
  *result.mutable_method() = std::move(*method);
  return result;
}

// Retrieves `QuantizationResult` from `xla_call_module_op`. If
// `xla_call_module_op` is a quantizable unit, then a `QuantizationResult` will
// be returned with its `name` field set to the callee's name. The `method`
// field will be set to `NoQuantization` because remaining `xla_call_module_op`s
// means they are not quantized. Returns `std::nullopt` if `xla_call_module_op`
// is not a quantizable unit.
std::optional<QuantizationResult> GetQuantizationResult(
    TF::XlaCallModuleOp xla_call_module_op) {
  const StringAttr callee_name_attr =
      mlir::dyn_cast_or_null<StringAttr>(xla_call_module_op->getDiscardableAttr(
          kOriginalStablehloEntryFunctionAttrName));

  // `TF::XlaCallModuleOp` without the `_original_entry_function` means it is
  // not a quantizable unit.
  if (callee_name_attr == nullptr) return std::nullopt;

  if (callee_name_attr.getValue().starts_with(kCompositeFuncPrefix)) {
    QuantizationResult result{};
    result.mutable_quantizable_unit()->set_name(
        callee_name_attr.getValue().str());
    result.mutable_method()->mutable_no_quantization();
    return result;
  } else {
    return std::nullopt;
  }
}

// Populates quantized ops from `module_op` to `results`. After going through
// the quantization passes, quantized ops are represented as `func::CallOp` with
// a callee's prefix of `quantized_`.
void PopulateQuantizedResults(ModuleOp module_op,
                              QuantizationResults& results) {
  module_op.walk([&results](func::CallOp call_op) {
    std::optional<QuantizationResult> result = GetQuantizationResult(call_op);
    if (result == std::nullopt) return WalkResult::skip();

    *results.add_results() = std::move(*result);
    return WalkResult::advance();
  });
}

// Populates non-quantized ops from `module_op` to `results`. After going
// through the quantization passes, non-quantized quantizable units remain as
// `TF::XlaCallModuleOp` with a callee's prefix of `composite_`.
void PopulateNonQuantizedResults(ModuleOp module_op,
                                 QuantizationResults& results) {
  module_op.walk([&results](TF::XlaCallModuleOp xla_call_module_op) {
    std::optional<QuantizationResult> result =
        GetQuantizationResult(xla_call_module_op);
    if (result == std::nullopt) return WalkResult::skip();

    *results.add_results() = std::move(*result);
    return WalkResult::advance();
  });
}

}  // namespace

QuantizationReport::QuantizationReport(ModuleOp module_op)
    : quantization_results_(CollectResultsFromModuleOp(module_op)) {}

QuantizationResults QuantizationReport::CollectResultsFromModuleOp(
    ModuleOp module_op) const {
  QuantizationResults results{};

  PopulateQuantizedResults(module_op, results);
  PopulateNonQuantizedResults(module_op, results);

  return results;
}

void QuantizationReport::AddQuantizationResult(QuantizationResult&& result) {
  *quantization_results_.add_results() = std::move(result);
}

std::string QuantizationReport::ToString() const {
  std::string results_str{};
  TextFormat::PrintToString(quantization_results_, &results_str);

  return absl::StrCat("===== Quantization Report =====\n\n", results_str,
                      "\n===== Quantization Report End =====\n\n");
}

void QuantizationReport::Print() const {
  llvm::outs() << ToString();
  llvm::outs().flush();  // Show the report immediately.
}

absl::Status QuantizationReport::Save(const StringRef file_path) const {
  std::string results_str{};
  TextFormat::PrintToString(GetQuantizationResults(), &results_str);

  return WriteStringToFile(file_path, results_str);
}

}  // namespace mlir::quant::stablehlo
