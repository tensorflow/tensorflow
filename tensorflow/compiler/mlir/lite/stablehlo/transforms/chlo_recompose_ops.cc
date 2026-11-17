/* Copyright 2024 The StableHLO Authors.

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

// Recomposes CHLO ops that were serialized as `stablehlo.composite` ops.
//
// Some CHLO ops have no StableHLO equivalent, so serializers that emit
// portable StableHLO (e.g. JAX's `jax.export`) wrap them in a composite named
// after the op, with the op's attributes stored as builtin attributes and the
// op itself as the decomposition:
//
//   %0 = stablehlo.composite "chlo.erf" %arg0 {
//          decomposition = @chlo.erf.impl, version = 1 : i32}
//
// This pass turns such composites back into the CHLO op, so that they lower
// the same way as CHLO ops emitted directly, e.g. `chlo.top_k` to
// `tfl.topk_v2`.
//
// Forked from the composite patterns of the `ChloRecomposeOpsPass` in XLA's
// StableHLO extensions. The deprecated `stablehlo.custom_call @mhlo.<op>`
// encodings are not handled.

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"  // IWYU pragma: keep

namespace mlir::odml {

#define GEN_PASS_DEF_CHLORECOMPOSEOPSPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

// ragged_dot_dimension_numbers
//   [[lhs_batch], [rhs_batch], [lhs_contract],
//    [rhs_contract], [lhs_ragged], [rhs_group]]
//   ==>
//   #chlo.ragged_dot<lhs_batch = [0], rhs_batch = [1], lhs_contract = [2],
//                    rhs_contract = [2], lhs_ragged = [1], rhs_group = [0]>
FailureOr<Attribute> DeserializeRaggedDotDimensionNumbersAttr(
    Operation* op, NamedAttribute attr) {
  auto array_attr = llvm::dyn_cast<ArrayAttr>(attr.getValue());
  if (!array_attr || array_attr.size() != 6) {
    return op->emitError() << "ragged_dot_dimension_numbers is not an "
                              "ArrayAttr with 6 elements";
  }
  auto lhs_batch = llvm::dyn_cast<DenseIntElementsAttr>(array_attr[0]);
  auto rhs_batch = llvm::dyn_cast<DenseIntElementsAttr>(array_attr[1]);
  auto lhs_contract = llvm::dyn_cast<DenseIntElementsAttr>(array_attr[2]);
  auto rhs_contract = llvm::dyn_cast<DenseIntElementsAttr>(array_attr[3]);
  auto lhs_ragged = llvm::dyn_cast<DenseIntElementsAttr>(array_attr[4]);
  auto rhs_group = llvm::dyn_cast<DenseIntElementsAttr>(array_attr[5]);
  if (!lhs_batch || !rhs_batch || !lhs_contract || !rhs_contract ||
      !lhs_ragged || !rhs_group) {
    return op->emitError() << "elements in ragged_dot_dimension_numbers are "
                              "not DenseIntElementsAttrs";
  }
  return chlo::RaggedDotDimensionNumbersAttr::get(
      op->getContext(), llvm::to_vector(lhs_batch.getValues<int64_t>()),
      llvm::to_vector(rhs_batch.getValues<int64_t>()),
      llvm::to_vector(lhs_contract.getValues<int64_t>()),
      llvm::to_vector(rhs_contract.getValues<int64_t>()),
      llvm::to_vector(lhs_ragged.getValues<int64_t>()),
      llvm::to_vector(rhs_group.getValues<int64_t>()));
}

// precision_config
//   ["DEFAULT", "DEFAULT"]
//   ==>
//   [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
FailureOr<Attribute> DeserializePrecisionConfigAttr(Operation* op,
                                                    NamedAttribute attr) {
  auto array_attr = llvm::dyn_cast<ArrayAttr>(attr.getValue());
  // Unlike the other malformed cases, this one emits no error (as in XLA).
  if (!array_attr) return failure();
  SmallVector<Attribute> chlo_attrs;
  for (Attribute precision_attr : array_attr) {
    auto precision_str = llvm::dyn_cast<StringAttr>(precision_attr);
    if (!precision_str) {
      return op->emitError()
             << "precision_config is not an ArrayAttr of StringAttrs";
    }
    std::optional<chlo::Precision> precision =
        chlo::symbolizePrecision(precision_str.getValue());
    if (!precision.has_value()) {
      return op->emitError("invalid precision string");
    }
    chlo_attrs.push_back(
        chlo::PrecisionAttr::get(op->getContext(), *precision));
  }
  return ArrayAttr::get(op->getContext(), chlo_attrs);
}

// Converts an attribute serialized as a builtin attribute back to the CHLO
// attribute. CHLO attributes cannot appear in VHLO, so serializers store them
// as builtin attributes.
FailureOr<Attribute> DeserializeChloAttribute(Operation* op, StringRef op_name,
                                              NamedAttribute attr) {
  if (op_name == chlo::RaggedDotOp::getOperationName()) {
    if (attr.getName() == "ragged_dot_dimension_numbers") {
      return DeserializeRaggedDotDimensionNumbersAttr(op, attr);
    }
    if (attr.getName() == "precision_config") {
      return DeserializePrecisionConfigAttr(op, attr);
    }
  }

  // Only builtin attributes pass through unchanged.
  if (attr.getValue().getDialect().getNamespace() != "builtin") {
    return op->emitError() << "unsupported attribute for chlo recompose:"
                           << attr.getValue();
  }
  return attr.getValue();
}

FailureOr<SmallVector<NamedAttribute>> DeserializeChloAttributes(
    Operation* op, StringRef op_name, DictionaryAttr attrs) {
  SmallVector<NamedAttribute> new_attrs;
  for (NamedAttribute attr : attrs.getValue()) {
    FailureOr<Attribute> chlo_attr =
        DeserializeChloAttribute(op, op_name, attr);
    if (failed(chlo_attr)) return failure();
    new_attrs.push_back({attr.getName(), *chlo_attr});
  }
  return new_attrs;
}

// Replaces `stablehlo.composite "<ChloOpTy name>"` (version 1) with ChloOpTy.
template <typename ChloOpTy>
struct ChloOpRecomposePattern
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getName() != ChloOpTy::getOperationName()) {
      return rewriter.notifyMatchFailure(op, "different op name");
    }
    if (op.getVersion() != 1) {
      return rewriter.notifyMatchFailure(op, "different version");
    }
    FailureOr<SmallVector<NamedAttribute>> attrs = DeserializeChloAttributes(
        op, op.getName(), op.getCompositeAttributes());
    if (failed(attrs)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to deserialize attributes");
    }

    ChloOpTy chlo_op;
    if constexpr (ChloOpTy::template hasTrait<OpTrait::VariadicRegions>()) {
      chlo_op = ChloOpTy::create(rewriter, op.getLoc(), op.getResultTypes(),
                                 op.getOperands(), *attrs, op.getNumRegions());
    } else {
      chlo_op = ChloOpTy::create(rewriter, op.getLoc(), op.getResultTypes(),
                                 op.getOperands(), *attrs);
    }
    if (op->getNumRegions() != chlo_op->getNumRegions()) {
      rewriter.eraseOp(chlo_op);
      return rewriter.notifyMatchFailure(op, "different number of regions");
    }
    for (auto [src_region, dst_region] :
         llvm::zip_equal(op.getCompositeRegions(), chlo_op->getRegions())) {
      rewriter.inlineRegionBefore(src_region, dst_region, dst_region.end());
    }
    rewriter.replaceOp(op, chlo_op);
    return success();
  }
};

class ChloRecomposeOpsPass
    : public impl::ChloRecomposeOpsPassBase<ChloRecomposeOpsPass> {
 public:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // The ops that serializers preserve as composites. Keep in sync with
    // XLA's `ChloPreserveHighLevelOpsPass`.
    patterns.add<ChloOpRecomposePattern<chlo::AcosOp>,
                 ChloOpRecomposePattern<chlo::AcoshOp>,
                 ChloOpRecomposePattern<chlo::AsinOp>,
                 ChloOpRecomposePattern<chlo::AsinhOp>,
                 ChloOpRecomposePattern<chlo::AtanhOp>,
                 ChloOpRecomposePattern<chlo::CoshOp>,
                 ChloOpRecomposePattern<chlo::SinhOp>,
                 ChloOpRecomposePattern<chlo::ErfOp>,
                 ChloOpRecomposePattern<chlo::ErfcOp>,
                 ChloOpRecomposePattern<chlo::MulhiOp>,
                 ChloOpRecomposePattern<chlo::RaggedDotOp>,
                 ChloOpRecomposePattern<chlo::ScanOp>,
                 ChloOpRecomposePattern<chlo::TopKOp>>(ctx);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

static PassRegistration<ChloRecomposeOpsPass> pass;

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateChloRecomposeOpsPass() {
  return std::make_unique<ChloRecomposeOpsPass>();
}

}  // namespace mlir::odml
