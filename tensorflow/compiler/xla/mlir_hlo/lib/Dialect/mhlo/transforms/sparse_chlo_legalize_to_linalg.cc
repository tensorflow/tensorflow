/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for converting CHLO dialect to Linalg dialect.

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/transforms/legalize_to_linalg_utils.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOLINALGPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct ChloLegalizeToLinalgPass
    : public impl::ChloLegalizeToLinalgPassBase<ChloLegalizeToLinalgPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                tensor::TensorDialect, sparse_tensor::SparseTensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    mhlo::RemoveSignTypeConverter typeConverter;
    mhlo::populateLegalizeSparseChloToLinalgPatterns(ctx, typeConverter,
                                                     &patterns);
    target.addLegalDialect<bufferization::BufferizationDialect,
                           linalg::LinalgDialect, tensor::TensorDialect,
                           sparse_tensor::SparseTensorDialect>();
    target.addIllegalDialect<chlo::ChloDialect>();
    /// The unary operation is sparse computation if either the input or the
    /// result is a sparse tensor.
    /// TODO(bixia): Remove the convert of such sparse CHLO ops from
    /// chlo_legalize_to_hlo.
    auto isNotSparseOp = [](Operation* op) {
      auto encDst =
          sparse_tensor::getSparseTensorEncoding(op->getResult(0).getType());
      auto encSrc =
          sparse_tensor::getSparseTensorEncoding(op->getOperand(0).getType());
      return !encDst && !encSrc;
    };
    target.addDynamicallyLegalOp<chlo::AsinOp, chlo::AsinhOp, chlo::AtanOp,
                                 chlo::AtanhOp, chlo::BesselI1eOp, chlo::SinhOp,
                                 chlo::TanOp>(isNotSparseOp);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

namespace impl {
/// Converts unary chlo op to a scalar op.
///
/// Since the CHLO ops require tensor operands, we first create a single element
/// from the tensor, then perform the CHLO ops, and extract the scalar result
/// from the tensor. This may introduce memory accesses overhead.
/// TODO(bixia): Remove the extra memory accesses for performance.
#define ADD_OP(OpTy)                                                           \
  template <>                                                                  \
  Value mapMhloOpToStdScalarOp<OpTy>(Location loc, ArrayRef<Type> resultTypes, \
                                     ArrayRef<Type> /*arg_types*/,             \
                                     ValueRange args, OpBuilder * b) {         \
    Type innerResultTy = resultTypes[0];                                       \
    RankedTensorType tensorResultTy =                                          \
        RankedTensorType::get({}, innerResultTy);                              \
    Value tensorArg =                                                          \
        b->create<tensor::FromElementsOp>(loc, tensorResultTy, args[0]);       \
    Value tensorResult =                                                       \
        b->create<OpTy>(loc, tensorResultTy, ValueRange({tensorArg}));         \
    Value innerResult =                                                        \
        b->create<tensor::ExtractOp>(loc, tensorResult, ValueRange({}));       \
    return innerResult;                                                        \
  }

ADD_OP(chlo::AsinOp)
ADD_OP(chlo::AsinhOp)
ADD_OP(chlo::AtanOp)
ADD_OP(chlo::AtanhOp)
ADD_OP(chlo::BesselI1eOp)
ADD_OP(chlo::SinhOp)
ADD_OP(chlo::TanOp)

#undef ADD_OP

}  // namespace impl

void populateLegalizeSparseChloToLinalgPatterns(MLIRContext* context,
                                                TypeConverter& typeConverter,
                                                RewritePatternSet* patterns) {
  patterns->add<PointwiseToLinalgConverter<chlo::AsinOp>,
                PointwiseToLinalgConverter<chlo::AsinhOp>,
                PointwiseToLinalgConverter<chlo::AtanOp>,
                PointwiseToLinalgConverter<chlo::AtanhOp>,
                PointwiseToLinalgConverter<chlo::SinhOp>,
                PointwiseToLinalgConverter<chlo::TanOp>,
                PointwiseToLinalgConverter<chlo::BesselI1eOp>>(typeConverter,
                                                               context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeSparseChloToLinalgPass() {
  return std::make_unique<ChloLegalizeToLinalgPass>();
}

}  // namespace mhlo

}  // namespace mlir
