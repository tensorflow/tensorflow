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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"

namespace mlir {
namespace mhlo {
namespace {

class LegalizeTF : public LegalizeTFBase<LegalizeTF> {
 public:
  explicit LegalizeTF(bool allow_partial_conversion, bool legalize_chlo,
                      llvm::Optional<StringRef> tf2xla_fallback_device_type,
                      bool prefer_tf2xla) {
    allow_partial_conversion_ = allow_partial_conversion;
    legalize_chlo_ = legalize_chlo;
    prefer_tf2xla_ = prefer_tf2xla;
    use_tf2xla_fallback_ = tf2xla_fallback_device_type.hasValue();
    if (tf2xla_fallback_device_type.hasValue()) {
      device_type_ = tf2xla_fallback_device_type.getValue().str();
    }
  }
  /// Performs the lowering to XLA dialect.
  void runOnOperation() override;
};

// Emits debug information which includes the number of ops of each type which
// failed to legalize.
void EmitLegalizationErrors(Operation *op,
                            const DenseSet<Operation *> &nonlegalized_ops) {
  // Track the legalization failures by mapping op name to information about
  // that failure: the number of unlegalized occurrences of the op, and one
  // example operation that failed.
  std::map<StringRef, std::pair<int, Operation *>> op_name_to_error_info;
  DenseSet<Operation *> error_ops;
  for (Operation *nonlegalized_op : nonlegalized_ops) {
    // Increment count of this legalization failure.
    StringRef op_name = nonlegalized_op->getName().getStringRef();
    // If this emplace is successful, it's the first time we've encountered
    // this op type. Initialize count to 0 so that after increment, it is 1.
    auto insertion_result = op_name_to_error_info.emplace(
        op_name, std::make_pair(0, nonlegalized_op));
    ++insertion_result.first->second.first;
  }
  std::vector<std::string> error_messages;
  error_messages.reserve(op_name_to_error_info.size());
  for (const auto &op_info : op_name_to_error_info) {
    error_messages.push_back(
        llvm::formatv("{0} (count: {1})", op_info.first, op_info.second.first));
  }
  Location loc = op->getLoc();
  emitError(loc) << "The following operations cannot be legalized: "
                 << llvm::join(error_messages, "; ")
                 << ". These legalization failure(s) may be due to missing TF "
                    "to HLO lowerings and/or unsupported attributes, etc.";
  // Emit more information about the missing ops. This error message
  // contains useful details beyond the op name (input and output shapes,
  // attributes, etc.).
  if (!VLOG_IS_ON(1) && nonlegalized_ops.size() != 1) {
    emitError(loc)
        << "Emitting more detail about one op that failed to legalize...";
  } else if (VLOG_IS_ON(1)) {
    emitError(loc) << "Emitting more detail about one of each type of op "
                      "that failed to legalize...";
  }
  for (const auto &op_info : op_name_to_error_info) {
    op_info.second.second->emitOpError() << "is not legalizable";
    if (!VLOG_IS_ON(1)) break;
  }
}

/// Returns ops that should use MLIR legalization only in the case of
/// prefer_tf2xla. All other ops not in this list should use XlaOpKernel
/// legalization only or not be legalized by the new bridge.
const llvm::DenseSet<mlir::TypeID> &MlirPreferredOps() {
  // The static variable is a pointer in order to avoid destruction upon thread
  // termination.

  // clang-format off
  static const llvm::DenseSet<mlir::TypeID>* ops =
      new llvm::DenseSet<mlir::TypeID>{
    // Ops that are legalized in the old bridge using MlirXlaOpKernel
    TypeID::get<TF::AbsOp>(),
    TypeID::get<TF::AtanOp>(),
    TypeID::get<TF::AvgPool3DOp>(),
    TypeID::get<TF::BiasAddGradOp>(),
    TypeID::get<TF::CeilOp>(),
    TypeID::get<TF::CheckNumericsOp>(),
    TypeID::get<TF::ComplexOp>(),
    TypeID::get<TF::CosOp>(),
    TypeID::get<TF::DiagPartOp>(),
    TypeID::get<TF::DivOp>(),
    TypeID::get<TF::EinsumOp>(),
    TypeID::get<TF::ExpOp>(),
    TypeID::get<TF::Expm1Op>(),
    TypeID::get<TF::FakeQuantWithMinMaxArgsOp>(),
    TypeID::get<TF::FloorOp>(),
    TypeID::get<TF::GreaterEqualOp>(),
    TypeID::get<TF::IFFTOp>(),
    TypeID::get<TF::ImagOp>(),
    TypeID::get<TF::IsFiniteOp>(),
    TypeID::get<TF::IsInfOp>(),
    TypeID::get<TF::IsNanOp>(),
    TypeID::get<TF::LessEqualOp>(),
    TypeID::get<TF::LgammaOp>(),
    TypeID::get<TF::Log1pOp>(),
    TypeID::get<TF::LogicalOrOp>(),
    TypeID::get<TF::LogSoftmaxOp>(),
    TypeID::get<TF::MatrixBandPartOp>(),
    TypeID::get<TF::MaxPool3DGradOp>(),
    TypeID::get<TF::PreventGradientOp>(),
    TypeID::get<TF::RandomShuffleOp>(),
    TypeID::get<TF::RealOp>(),
    TypeID::get<TF::ReciprocalOp>(),
    TypeID::get<TF::ReluOp>(),
    TypeID::get<TF::Relu6Op>(),
    TypeID::get<TF::ReluGradOp>(),
    TypeID::get<TF::RsqrtOp>(),
    TypeID::get<TF::SelectOp>(),
    TypeID::get<TF::SigmoidOp>(),
    TypeID::get<TF::SignOp>(),
    TypeID::get<TF::SoftmaxOp>(),
    TypeID::get<TF::SqrtOp>(),
    TypeID::get<TF::SqrtGradOp>(),
    TypeID::get<TF::SquaredDifferenceOp>(),
    TypeID::get<TF::TanhOp>(),
    TypeID::get<TF::TanhGradOp>(),
    TypeID::get<TF::XlaDotOp>(),
    TypeID::get<TF::XlaDotV2Op>(),
    TypeID::get<TF::XlaDynamicSliceOp>(),
    TypeID::get<TF::XlaEinsumOp>(),
    TypeID::get<TF::XlaReduceWindowOp>(),
    TypeID::get<TF::XlaReplicaIdOp>(),
    TypeID::get<TF::XlaRngBitGeneratorOp>(),
    TypeID::get<TF::XlaSelectAndScatterOp>(),
    TypeID::get<TF::XlaSortOp>(),
    TypeID::get<TF::XlaVariadicReduceV2Op>(),
    TypeID::get<TF::XlaVariadicSortOp>(),
    TypeID::get<TF::XlogyOp>(),
    TypeID::get<TF::ZetaOp>(),

    // Ops that have no XlaOpKernel.
    TypeID::get<TF::RiscAddOp>(),
    TypeID::get<TF::RiscDotOp>(),

    // Const op has a simple legalization and it is much more efficient to lower
    // within MLIR.
    TypeID::get<TF::ConstOp>(),

    // AssertOp with string types are not supported by the fallback.
    TypeID::get<TF::AssertOp>(),

    // TF2XLA fallback pattern doesn't support these op as MLIR hlo builder
    // doesn't override the necessary builder methods. These ops have simple
    // lowering pattern so this should be safe.
    TypeID::get<TF::CrossReplicaSumOp>(),
    TypeID::get<TF::InfeedDequeueTupleOp>(),
    TypeID::get<TF::OutfeedEnqueueTupleOp>(),
    TypeID::get<TF::XlaShardingOp>(),

    // These ops have undetermined bugs, may not be legalizable with XlaOpKernel
    // legalization in TF2XLA fallback. By legalization with MLIR, we can fix
    // the bug. b/195583695 describes the motivation of this change.
    // See b/216355804 how to reproduce the bug regarding tf.RandomUniform Op
    // See b/216353817 how to reproduce the bug regarding tf.StridedSlice Op
    TypeID::get<TF::RandomUniformOp>(),
    TypeID::get<TF::StridedSliceOp>(),
  };
  // clang-format on
  return *ops;
}

// Patterns whose root op is in the set `include_ops` are moved from the set
// `from` to the returned set. This is used to partition patterns by op so they
// can be cleanly migrated from the old bridge to the MLIR bridge.
RewritePatternSet PatternsIncludeOps(
    RewritePatternSet &from, const llvm::DenseSet<mlir::TypeID> &include_ops) {
  RewritePatternSet to(from.getContext());
  // Filter NativePatterns.
  for (auto &pattern : from.getNativePatterns()) {
    Optional<OperationName> pat_op_name = pattern->getRootKind();
    // If the pattern does not have a specific operation, always include it,
    // If the pattern is in include_ops then include it.
    bool include =
        !pat_op_name ||
        include_ops.count(pat_op_name->getRegisteredInfo()->getTypeID());
    if (include) to.add(std::move(pattern));
  }

  // Don't filter PDLPatterns.
  to.add(std::move(from.getPDLPatterns()));

  return to;
}

/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
LogicalResult legalizeTF(Operation *op, bool allow_partial_conversion,
                         bool legalize_chlo,
                         llvm::Optional<StringRef> tf2xla_fallback_device_type,
                         bool prefer_tf2xla) {
  MLIRContext *context = op->getContext();
  RewritePatternSet legalize_lower_patterns(context);
  // Note that the `OperationConverter` orders patterns lexicographically by:
  // 1) Ascending legalization depth (i.e., minimum number of patterns necessary
  //    to arrive at conversion target). This requires relevant patterns to
  //    specify the list of ops generated by it which most of patterns
  //    implemented in C++ don't do so this comparison doesn't work in those
  //    cases.
  // 2) Descending pattern benefit.
  // 3) Op specific patterns over patterns with MatchAnyOpTypeTag.
  // 4) Order of patterns in `RewritePatternSet`.

  // Add TF->HLO legalization patterns.
  PopulateLegalizeTfPatterns(context, &legalize_lower_patterns);

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

  // Set patterns to legalize_lower_patters, where in the prefer_tf2xla case
  // only patterns whose ops are in the set MlirPreferredOps are kept.
  RewritePatternSet patterns =
      (tf2xla_fallback_device_type && prefer_tf2xla)
          ? PatternsIncludeOps(legalize_lower_patterns, MlirPreferredOps())
          : std::move(legalize_lower_patterns);

  if (tf2xla_fallback_device_type) {
    // Add TF->HLO legalization patterns via TF2XLA fallback.
    PopulateLegalizeTfWithTf2XlaPatterns(tf2xla_fallback_device_type.getValue(),
                                         patterns, context, prefer_tf2xla);
  }

  // Populate with CHLO->HLO lowerings to account for TF ops legalized to
  // CHLO first.
  if (legalize_chlo) {
    chlo::populateDecomposeChloPatterns(context, &patterns);
    chlo::populateChloBroadcastingPatterns(context, &patterns);
  }
  // ConstantLike op is convenient to create splat constants, but is
  // canonicalized to plain HLO constant if statically shaped. Add the
  // canonicalization pattern to pattern list to enable multi-hop lowering.
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  ConversionTarget target(*context);
  if (legalize_chlo) {
    target.addIllegalDialect<chlo::ChloDialect>();
  } else {
    target.addLegalDialect<chlo::ChloDialect>();
  }
  target.addLegalDialect<MhloDialect>();
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  if (!allow_partial_conversion) {
    // Fully qualify ReturnOp here as mhlo dialect also defines a ReturnOp.
    target.addLegalOp<ModuleOp, ::mlir::func::FuncOp, ::mlir::func::ReturnOp>();
    DenseSet<Operation *> nonlegalized_ops;
    LogicalResult result = applyPartialConversion(
        op, target, std::move(patterns), &nonlegalized_ops);
    // In order to enforce that the conversion result is fully converted,
    // fail if there are any nonlegalized ops in the set.
    if (failed(result) || !nonlegalized_ops.empty()) {
      EmitLegalizationErrors(op, nonlegalized_ops);
      return failure();
    }
    return result;
  }

  return applyPartialConversion(op, target, std::move(patterns));
}

// Performs the lowering to XLA dialect.
void LegalizeTF::runOnOperation() {
  llvm::Optional<StringRef> tf2xla_fallback_device_type = llvm::None;
  if (use_tf2xla_fallback_) {
    tf2xla_fallback_device_type = device_type_;
  }
  if (failed(legalizeTF(getOperation(), allow_partial_conversion_,
                        legalize_chlo_, tf2xla_fallback_device_type,
                        prefer_tf2xla_))) {
    signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion, bool legalize_chlo,
    llvm::Optional<StringRef> tf2xla_fallback_device_type, bool prefer_tf2xla) {
  return std::make_unique<LegalizeTF>(allow_partial_conversion, legalize_chlo,
                                      tf2xla_fallback_device_type,
                                      prefer_tf2xla);
}

}  // end namespace mhlo
}  // end namespace mlir
