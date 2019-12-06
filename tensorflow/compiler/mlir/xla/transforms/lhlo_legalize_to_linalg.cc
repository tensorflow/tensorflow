/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include "absl/memory/memory.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // TF:local_config_mlir
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Transforms/DialectConversion.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {
namespace {

// TODO(pifon): Move LHLO -> STD op map to a separate lib.
template <typename LHLO_BinaryOp>
struct ScalarOp;

template <>
struct ScalarOp<xla_lhlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct ScalarOp<xla_lhlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::DivISOp;
};
template <>
struct ScalarOp<xla_lhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct ScalarOp<xla_lhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};
template <typename LHLO_BinaryOp>
using ScalarFOp = typename ScalarOp<LHLO_BinaryOp>::FOp;
template <typename LHLO_BinaryOp>
using ScalarIOp = typename ScalarOp<LHLO_BinaryOp>::IOp;

template <typename LhloOp>
Operation* GetLinalgBodyOp(Location loc, Type element_type,
                           ArrayRef<Type> body_result_types,
                           ArrayRef<Value*> block_args, OpBuilder b) {
  if (element_type.isa<IntegerType>()) {
    return b.template create<ScalarIOp<LhloOp>>(loc, body_result_types,
                                                block_args, mlir::None);
  }
  if (element_type.isa<FloatType>()) {
    return b.template create<ScalarFOp<LhloOp>>(loc, body_result_types,
                                                block_args, mlir::None);
  }
  return nullptr;
}

template <>
Operation* GetLinalgBodyOp<xla_lhlo::MaxOp>(Location loc, Type element_type,
                                            ArrayRef<Type> body_result_types,
                                            ArrayRef<Value*> block_args,
                                            OpBuilder b) {
  const auto& lhs = block_args[0];
  const auto& rhs = block_args[1];
  if (element_type.isa<IntegerType>()) {
    auto lhs_gt_rhs = b.create<CmpIOp>(loc, CmpIPredicate::SGT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_gt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_gt_rhs = b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_gt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
Operation* GetLinalgBodyOp<xla_lhlo::MinOp>(Location loc, Type element_type,
                                            ArrayRef<Type> body_result_types,
                                            ArrayRef<Value*> block_args,
                                            OpBuilder b) {
  const auto& lhs = block_args[0];
  const auto& rhs = block_args[1];
  if (element_type.isa<IntegerType>()) {
    auto lhs_lt_rhs = b.create<CmpIOp>(loc, CmpIPredicate::SLT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_lt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_lt_rhs = b.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_lt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
Operation* GetLinalgBodyOp<xla_lhlo::AndOp>(Location loc, Type element_type,
                                            ArrayRef<Type> body_result_types,
                                            ArrayRef<Value*> block_args,
                                            OpBuilder b) {
  return element_type.isa<IntegerType>()
             ? b.create<::mlir::AndOp>(loc, body_result_types, block_args,
                                       mlir::None)
             : nullptr;
}

template <typename LhloOp>
class LhloToLinalgOpConverter : public ConversionPattern {
 public:
  explicit LhloToLinalgOpConverter(MLIRContext* context)
      : ConversionPattern(LhloOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation* lhlo_op, ArrayRef<Value*> args,
      ConversionPatternRewriter& rewriter) const final {
    const auto& loc = lhlo_op->getLoc();
    auto arg_type = lhlo_op->getOperand(0)->getType().dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasStaticShape()) {
      emitError(loc,
                "lhlo to linalg conversion expects statically shaped args");
      return matchFailure();
    }
    if (!arg_type || !arg_type.getElementType().isIntOrFloat()) {
      return matchFailure();
    }

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Attribute, 2> indexing_maps;
    SmallVector<Type, 4> body_arg_types, body_result_types;
    unsigned nloops = 0;
    for (const auto& arg : llvm::enumerate(args)) {
      auto memref_type = arg.value()->getType().dyn_cast<MemRefType>();
      if (!memref_type) {
        return matchFailure();
      }
      if (nloops && nloops != memref_type.getRank()) {
        return matchFailure();
      }
      nloops = std::max(nloops, static_cast<unsigned>(memref_type.getRank()));
      indexing_maps.emplace_back(
          rewriter.getAffineMapAttr(rewriter.getMultiDimIdentityMap(nloops)));
      auto& result_or_body_arg =
          arg.index() < 2 ? body_arg_types : body_result_types;
      result_or_body_arg.emplace_back(memref_type.getElementType());
    }

    // Pointwise-ops have all surrounding loops parallel, so the loop triple is
    // [argDim, 0, 0].
    const SmallVector<Attribute, 3> loop_types{
        rewriter.getI64IntegerAttr(nloops), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0)};
    // Define the number of input memref/output memrefs.
    const SmallVector<Attribute, 2> nmemrefs{
        rewriter.getI64IntegerAttr(body_arg_types.size()),
        rewriter.getI64IntegerAttr(body_result_types.size())};

    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, args, rewriter.getArrayAttr(indexing_maps),
        rewriter.getArrayAttr(loop_types), rewriter.getArrayAttr(nmemrefs),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalg_op.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(body_arg_types);
    block->addArguments(body_result_types);

    SmallVector<Value*, 4> body_args;
    for (int i = 0, e = body_arg_types.size(); i < e; ++i) {
      body_args.push_back(block->getArgument(i));
    }

    rewriter.setInsertionPointToEnd(block);
    Operation* op = GetLinalgBodyOp<LhloOp>(
        loc, body_arg_types[0], body_result_types, body_args, rewriter);
    rewriter.create<linalg::YieldOp>(loc, llvm::to_vector<1>(op->getResults()));
    rewriter.replaceOp(lhlo_op, {});
    return matchSuccess();
  }
};

void populateLHLOToLinalgConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  patterns->insert<LhloToLinalgOpConverter<xla_lhlo::AddOp>,
                   LhloToLinalgOpConverter<xla_lhlo::AndOp>,
                   LhloToLinalgOpConverter<xla_lhlo::DivOp>,
                   LhloToLinalgOpConverter<xla_lhlo::MaxOp>,
                   LhloToLinalgOpConverter<xla_lhlo::MinOp>,
                   LhloToLinalgOpConverter<xla_lhlo::MulOp>,
                   LhloToLinalgOpConverter<xla_lhlo::SubOp>>(context);
}

// Converts LHLO ops to Linalg generic.
// Sample result for xla_lhlo::AddOp.
//
// "xla_lhlo.add"(%arg1, %arg2, %out) :
//      (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//
// will be converted to
//
// #map0 = (d0, d1) -> (d0, d1)
// "linalg.generic"(%arg1, %arg2, %out) ( {
//   ^bb0(%arg4: f32, %arg5: f32):
//     %0 = addf %arg4, %arg5 : f32
//     "linalg.yield"(%0) : (f32) -> ()
//   }) {
//     indexing_maps = [#map0, #map0, #map0],
//     n_loop_types = [2, 0, 0],
//     n_views = [2, 1]
//   } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
// }
struct LhloLegalizeToLinalg : public FunctionPass<LhloLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    populateLHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToLinalgPass() {
  return absl::make_unique<LhloLegalizeToLinalg>();
}

static PassRegistration<LhloLegalizeToLinalg> legalize_pass(
    "lhlo-legalize-to-linalg", "Legalize from LHLO dialect to Linalg dialect");

}  // namespace xla_lhlo
}  // namespace mlir
