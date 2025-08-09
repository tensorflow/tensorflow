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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "mhlo/transforms/rewriters.h"
#include "absl/log/log.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalize_tf_with_tf2xla_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep, dependent dialect
#include "xla/mlir_hlo/mhlo/transforms/rewriters.h"
#include "xla/mlir_hlo/mhlo/utils/type_conversion.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_LEGALIZETF
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

auto *mlir_legalization_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_count",
    "Counts the attempts of legalization of ops", "op_name");

auto *mlir_failed_legalization_count = tensorflow::monitoring::Counter<2>::New(
    "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_pass_count",
    "Counts the failure of legalization of ops", "op_name", "legality");

class LegalizeTF : public impl::LegalizeTFBase<LegalizeTF> {
 public:
  explicit LegalizeTF(bool legalize_chlo,
                      std::optional<StringRef> tf2xla_fallback_device_type,
                      bool prefer_tf2xla) {
    legalize_chlo_ = legalize_chlo;
    prefer_tf2xla_ = prefer_tf2xla;
    use_tf2xla_fallback_ = tf2xla_fallback_device_type.has_value();
    if (tf2xla_fallback_device_type.has_value()) {
      device_type_ = tf2xla_fallback_device_type.value().str();
    }
  }
  /// Performs the lowering to XLA dialect.
  void runOnOperation() override;
};

#define GEN_PASS_DEF_LEGALIZETFMODULEPASS
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

// Patterns whose root op is in the set `include_ops` are moved from the set
// `from` to the returned set. This is used to partition patterns by op so they
// can be cleanly migrated from the old bridge to the MLIR bridge.
RewritePatternSet PatternsIncludeOps(RewritePatternSet &from) {
  RewritePatternSet to(from.getContext());
  // Filter NativePatterns.
  for (auto &pattern : from.getNativePatterns()) {
    std::optional<OperationName> pat_op_name = pattern->getRootKind();
    // If the pattern does not have a specific operation, always include it,
    // If the pattern is in include_ops then include it.
    bool include =
        !pat_op_name || hlo::IsTypeLegalizedWithMlir(
                            pat_op_name->getRegisteredInfo()->getTypeID());
    if (include) to.add(std::move(pattern));
  }

  // Don't filter PDLPatterns.
  to.add(std::move(from.getPDLPatterns()));

  return to;
}

std::string OperationLegalityString(Operation *op,
                                    const ConversionTarget &target) {
  auto op_name = op->getName();
  auto action = target.getOpAction(op_name);
  if (!action.has_value()) {
    return "Unknown";
  }
  switch (action.value_or(ConversionTarget::LegalizationAction::Legal)) {
    case ConversionTarget::LegalizationAction::Legal:
      return "Legal";
    case ConversionTarget::LegalizationAction::Dynamic:
      return "Dynamic";
    case ConversionTarget::LegalizationAction::Illegal:
      return "Illegal";
    default:
      return "Invalid";
  }
}

void IncrementFailedLegalizationCount(Operation *op,
                                      const ConversionTarget &target) {
  auto op_name = op->getName();
  auto name_string = op_name.getStringRef().str();
  auto op_legality = OperationLegalityString(op, target);

  mlir_failed_legalization_count->GetCell(name_string, op_legality)
      ->IncrementBy(1);
}

mlir::LogicalResult ApplyPatterns(Operation *op, RewritePatternSet &patterns,
                                  bool legalize_chlo) {
  ConversionTarget target =
      hlo::GetDefaultLegalConversionTargets(*op->getContext(), legalize_chlo);

  DenseSet<Operation *> unconverted_ops;
  ConversionConfig config;
  config.unlegalizedOps = &unconverted_ops;
  auto result = applyPartialConversion(op, target, std::move(patterns), config);
  if (failed(result)) {
    IncrementFailedLegalizationCount(op, target);
  }
  for (const auto &unconverted_op : unconverted_ops) {
    IncrementFailedLegalizationCount(unconverted_op, target);
  }
  return result;
}

mlir::LogicalResult StablehloToMhlo(Operation *op) {
  ConversionTarget target(*op->getContext());
  stablehlo::setupStablehloToHloConversionTarget(target);

  RewritePatternSet patterns(op->getContext());
  stablehlo::StablehloToHloTypeConverter shlo_converter;
  stablehlo::populateStablehloToHloPatterns(&patterns, &shlo_converter,
                                            patterns.getContext());
  stablehlo::registerFuncOpsForTypeConversion(target, patterns, shlo_converter);

  if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
    return op->emitError("TF2XLA failed to convert StableHLO to MHLO");
  }
  return success();
}

/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is
/// not used.
LogicalResult legalizeTF(Operation *op, bool legalize_chlo,
                         std::optional<StringRef> tf2xla_fallback_device_type,
                         bool prefer_tf2xla) {
  MLIRContext *context = op->getContext();
  RewritePatternSet legalize_lower_patterns(context);
  // Note that the `OperationConverter` orders patterns lexicographically by:
  // 1) Ascending legalization depth (i.e., minimum number of patterns
  // necessary
  //    to arrive at conversion target). This requires relevant patterns to
  //    specify the list of ops generated by it which most of patterns
  //    implemented in C++ don't do so this comparison doesn't work in those
  //    cases.
  // 2) Descending pattern benefit.
  // 3) Op specific patterns over patterns with MatchAnyOpTypeTag.
  // 4) Order of patterns in `RewritePatternSet`.

  // Add TF->HLO legalization patterns.
  hlo::PopulateLegalizeTfPatterns(context, &legalize_lower_patterns);

  // Add TF->TF lowering patterns.
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &legalize_lower_patterns);

  if (tf2xla_fallback_device_type && prefer_tf2xla) {
    VLOG(1) << "TF to XLA legalization patterns are partitioned by op into "
               "either native MLIR legalization, or TF2XLA fallback "
               "legalzation, with a preference toward TF2XLA.";
  } else if (tf2xla_fallback_device_type) {
    VLOG(1) << "TF to XLA legalization patterns include all native patterns "
               "and TF2XLA fallback patterns.";
  } else {
    VLOG(1) << "TF to XLA legalization patterns are native patterns only.";
  }

  // Set patterns to legalize_lower_patterns to check whether they should use
  // MLIR or TF2XLA lowering patterns.
  RewritePatternSet patterns = (tf2xla_fallback_device_type && prefer_tf2xla)
                                   ? PatternsIncludeOps(legalize_lower_patterns)
                                   : std::move(legalize_lower_patterns);

  Tf2XlaTypeConverter converter;
  if (tf2xla_fallback_device_type) {
    // Add TF->HLO legalization patterns via TF2XLA fallback.
    PopulateLegalizeTfWithTf2XlaPatterns(tf2xla_fallback_device_type.value(),
                                         patterns, context, converter,
                                         prefer_tf2xla);
  }

  // Populate with CHLO->HLO lowerings to account for TF ops legalized to
  // CHLO first.
  stablehlo::StablehloToHloTypeConverter hlo_converter;
  stablehlo::populateStablehloToHloPatterns(&patterns, &hlo_converter, context);
  if (legalize_chlo) {
    chlo::populateChloToHighLevelMhloOpPatterns(context, &patterns);
    stablehlo::populateChloToStablehloPatterns(context, &patterns);
  }
  // ConstantLike op is convenient to create splat constants, but is
  // canonicalized to plain HLO constant if statically shaped. Add the
  // canonicalization pattern to pattern list to enable multi-hop lowering.
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  if (failed(ApplyPatterns(op, patterns, legalize_chlo))) {
    return failure();
  }

  // HLO->MLIR raises to StableHLO, but users of this pass expect MHLO.
  return StablehloToMhlo(op);
}

// Performs the lowering to XLA dialect.
void LegalizeTF::runOnOperation() {
  auto op = getOperation();
  VLOG(3) << "LegalizeTF(legalize_chlo=" << legalize_chlo_
          << ", prefer_tf2xla=" << prefer_tf2xla_ << ") on module:\n"
          << mlir::debugString(*op);
  auto op_name = op->getName().getStringRef().str();
  mlir_legalization_count->GetCell(op_name)->IncrementBy(1);
  std::optional<StringRef> tf2xla_fallback_device_type = std::nullopt;
  if (use_tf2xla_fallback_) {
    tf2xla_fallback_device_type = device_type_;
  }
  if (failed(legalizeTF(op, legalize_chlo_, tf2xla_fallback_device_type,
                        prefer_tf2xla_))) {
    signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFPass(
    bool legalize_chlo, std::optional<StringRef> tf2xla_fallback_device_type,
    bool prefer_tf2xla) {
  return std::make_unique<LegalizeTF>(
      legalize_chlo, tf2xla_fallback_device_type, prefer_tf2xla);
}

}  // end namespace mhlo
}  // end namespace mlir
