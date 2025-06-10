/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/regexp.h"    // IWYU pragma: keep

#define DEBUG_TYPE "lift_quantizable_spots_as_functions"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_LIFTQUANTIZABLESPOTSASFUNCTIONSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using ::stablehlo::quantization::FunctionNameMatcherSpec;
using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::QuantizationSpec;
using ::stablehlo::quantization::QuantizationSpecs;
using ::tsl::protobuf::TextFormat;

using tf_quant::FunctionCallOpType;
using tf_quant::GetSortedFunctions;
using tf_quant::IsInLiftedFunc;
using tf_quant::IsInStableHloOpRegion;
using tf_quant::kAttrMapAttribute;
using tf_quant::kNullAttributeValue;
using tf_quant::kQuantizationMethodAttr;

// TODO - b/303543789: Move the helper functions below to a separate util.
// Fetches the default or null attribute, used for pattern matching.
Attribute DefaultOrNullAttr(OpBuilder& builder, const Attribute& attr) {
  if (attr) return attr;
  return builder.getStringAttr(kNullAttributeValue);
}

// Checks whether the value of a constant equals the given float, regardless
// of the tensor dimension.
bool FloatValueEquals(const Attribute& attr, const double value) {
  const auto fp_attr = mlir::dyn_cast_or_null<DenseFPElementsAttr>(attr);
  if (!fp_attr) return false;

  if (fp_attr.isSplat()) {
    return fp_attr.getSplatValue<APFloat>().isExactlyValue(value);
  }
  return llvm::all_of(fp_attr.getValues<APFloat>(), [value](const APFloat& f) {
    return f.isExactlyValue(value);
  });
}

inline void TrimTrailingWhitespaces(std::string& str) {
  while (!str.empty() && str.back() == ' ') {
    str.pop_back();
  }
}

// Lifts quantizable units as separate functions, thereby identifying the
// boundaries of quantizable subgraphs. `QuantizationSpecs` influences how
// quantizable units are lifted.
//
// FileCheck test cases using various `QuantizationSpecs` can be seen at
// `TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass`.
class LiftQuantizableSpotsAsFunctionsPass
    : public impl::LiftQuantizableSpotsAsFunctionsPassBase<
          LiftQuantizableSpotsAsFunctionsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LiftQuantizableSpotsAsFunctionsPass)

  LiftQuantizableSpotsAsFunctionsPass() = default;

  // Constructor with explicit user-provided `QuantizationSpecs`.
  explicit LiftQuantizableSpotsAsFunctionsPass(
      QuantizationSpecs quantization_specs)
      : quantization_specs_(std::move(quantization_specs)) {}

 private:
  void runOnOperation() override;

  // No explicit quantization spec is specified by default. Implicitly this
  // means that all quantizable units will be identified and lifted.
  QuantizationSpecs quantization_specs_{};
};

namespace simple_patterns {
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/lift_quantizable_spots_as_functions_simple.inc"
}

namespace fusion_patterns {
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/lift_quantizable_spots_as_functions_fusion.inc"
}

// Quantizable Unit matcher that uses lifted function's name for matching.
class FunctionNameMatcher {
 public:
  explicit FunctionNameMatcher(const FunctionNameMatcherSpec& spec)
      : match_regex_(GetMatchRegex(spec)) {}

  // Returns `true` when matched with the entry function of
  // `xla_call_module_op`.
  bool Match(TF::XlaCallModuleOp xla_call_module_op) const {
    if (match_regex_ == nullptr) return false;

    const std::string lifted_func_name =
        xla_call_module_op->getAttrOfType<FlatSymbolRefAttr>("_entry_function")
            .getValue()
            .str();

    return RE2::FullMatch(lifted_func_name, *match_regex_);  // NOLINT
  }

 private:
  // Returns an owned `RE2` object that corresponds to the `spec`. Returns
  // `nullptr` if the `spec` is invalid.
  // NOLINTNEXTLINE - RE2 included via TSL regexp.h
  std::unique_ptr<RE2> GetMatchRegex(const FunctionNameMatcherSpec& spec) {
    const std::string& regex = spec.regex();
    if (regex.empty()) return nullptr;

    return std::make_unique<RE2>(regex);  // NOLINT
  }

  // Regex object used for matching against a lifted function's name.
  std::unique_ptr<RE2> match_regex_;  // NOLINT
};

// Converts `Method` to a single-line textproto representation. Returns
// `failure()` when converting to textproto failed.
FailureOr<std::string> QuantizationMethodToTextProto(const Method& method) {
  TextFormat::Printer printer;
  printer.SetSingleLineMode(true);

  std::string method_txtpb;
  if (!printer.PrintToString(method, &method_txtpb)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to convert Method to textproto\n.");
    return failure();
  }

  // Single line mode might have an extra space at the end, due to the internal
  // details of `Printer`.
  TrimTrailingWhitespaces(method_txtpb);

  return method_txtpb;
}

// Applies quantization spec to all matched lifted functions. At this point only
// denylisting (`NoQuantization`) will be applied if specs is nonempty.
// TODO: b/307620778 - Support more advanced selective quantization methods.
LogicalResult ApplyQuantizationSpec(const QuantizationSpec& spec,
                                    ModuleOp module_op) {
  const Method& quantization_method = spec.method();

  FailureOr<std::string> quantization_method_txtpb =
      QuantizationMethodToTextProto(quantization_method);
  if (failed(quantization_method_txtpb)) return failure();

  const FunctionNameMatcher matcher(spec.matcher().function_name());
  // Iterate over all XlaCallModuleOp in all FuncOps.
  for (auto func : module_op.getOps<func::FuncOp>()) {
    for (auto xla_call_module_op : func.getOps<TF::XlaCallModuleOp>()) {
      if (!matcher.Match(xla_call_module_op)) continue;

      // Set the text representation of `Method` to matched
      // `TF::XlaCallModuleOp`.
      xla_call_module_op->setAttr(
          kQuantizationMethodAttr,
          StringAttr::get(module_op.getContext(),
                          std::move(*quantization_method_txtpb)));
    }
  }
  return success();
}

void LiftQuantizableSpotsAsFunctionsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module_op = getOperation();

  simple_patterns::populateWithGenerated(patterns);
  fusion_patterns::populateWithGenerated(patterns);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  // Iterate over the sorted list of functions to keep order deterministic.
  for (func::FuncOp func : GetSortedFunctions(module_op)) {
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError()
          << "quant-stablehlo-lift-quantizable-spots-as-functions failed.";
      signalPassFailure();
    }
  }

  // Remove all attr_map attributes.
  module_op.walk([](Operation* op) { op->removeAttr(kAttrMapAttribute); });

  // Perform selective quantization. Iterates over the quantization specs and
  // applies quantization methods to each matched lifted function.
  for (const QuantizationSpec& spec : quantization_specs_.specs()) {
    if (failed(ApplyQuantizationSpec(spec, module_op))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace

// Creates `LiftQuantizableSpotsAsFunctionsPass` with user-defined
// `QuantizationSpecs`.
std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsPass(
    const QuantizationSpecs& quantization_specs) {
  return std::make_unique<LiftQuantizableSpotsAsFunctionsPass>(
      quantization_specs);
}

}  // namespace mlir::quant::stablehlo
