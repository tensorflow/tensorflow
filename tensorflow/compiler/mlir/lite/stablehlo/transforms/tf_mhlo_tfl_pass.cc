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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_mhlo_tfl_pass.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/mhlo_util.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace TFL {
namespace mhlo {

namespace {

// Converts an `mhlo.compare` general Op to specific TFL comparison Ops such
// as `tfl.greater`. Needs to be instantiated with specific target types, and
// comparison directions.
//
// Example:
//   patterns.add<ConvertMhloCompareOp<mlir::TFL::GreaterEqualOp>>(
//       patterns.getContext(),
//       ::mlir::mhlo::ComparisonDirection::GE);
//
template <typename TFL_CompareOp, mlir::mhlo::ComparisonDirection Direction>
struct ConvertMhloCompareOp
    : public OpConversionPattern<::mlir::mhlo::CompareOp> {
 public:
  explicit ConvertMhloCompareOp(MLIRContext *context)
      : OpConversionPattern<::mlir::mhlo::CompareOp>(context) {}

  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::mhlo::CompareOp op, mlir::mhlo::CompareOp::Adaptor adaptor,
      ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto direction = op.comparison_direction();

    if (direction != Direction) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TFL_CompareOp>(op, adaptor.lhs(),
                                               adaptor.rhs());
    return success();
  }
};

// Convert the MHLO Atan2 Op which cannot be mapped to TFL directly to a
// `tfl.custom` Op which can be executed using custom TFL kernels.
struct ConvertMhloAtan2ToTflCustomOp
    : public OpConversionPattern<::mlir::mhlo::Atan2Op> {
 public:
  explicit ConvertMhloAtan2ToTflCustomOp(::mlir::MLIRContext *context)
      : OpConversionPattern<::mlir::mhlo::Atan2Op>(context) {}

  ConstBytesAttr BuildEmptyConstBytesAttr(Operation *op) const {
    OpBuilder builder(op);

    return ConstBytesAttr::get(builder.getContext(), StringRef());
  }

  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::mhlo::Atan2Op op, OpAdaptor adaptor,
      ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto op_code = op->getName().stripDialect().str();
    auto custom_option = BuildEmptyConstBytesAttr(op);

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(op, op->getResultTypes(),
                                                     adaptor.getOperands(),
                                                     op_code, custom_option);
    return success();
  }
};

}  // namespace

class TFMhloTFLPass
    : public mlir::PassWrapper<TFMhloTFLPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                    mlir::TFL::TensorFlowLiteDialect>();
  }

 public:
  StringRef getArgument() const final { return "tf-mhlo-tfl"; }
  StringRef getDescription() const final {
    return "This pass will legalize TF ops to TFL via mHLO.";
  }
};

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_mhlo_tfl_legalize_patterns.inc"

// Convert TF to MHLO and then to TFLite.
// This is "all or nothing" based conversion. If a TF op converts to a few MHLO
// ops that cannot fully convert to TFLite, the original TF op is kept.
void TFMhloTFLPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = func->getContext();

  RewritePatternSet patterns(context);

  // Add TF to MHLO patterns.
  PopulateTFToMhloPatterns(
      context, /*legalize_chlo=*/true,
      /*tf2xla_fallback_device_type=*/llvm::StringRef("XLA_CPU_JIT"),
      /*prefer_tf2xla=*/false, &patterns);

  // Add MHLO to TFL patterns.
  populateWithGenerated(patterns);
  patterns.add<ConvertMhloAtan2ToTflCustomOp,
               ConvertMhloCompareOp<mlir::TFL::NotEqualOp,
                                    ::mlir::mhlo::ComparisonDirection::NE>,
               ConvertMhloCompareOp<mlir::TFL::LessOp,
                                    ::mlir::mhlo::ComparisonDirection::LT>,
               ConvertMhloCompareOp<mlir::TFL::LessEqualOp,
                                    ::mlir::mhlo::ComparisonDirection::LE>,
               ConvertMhloCompareOp<mlir::TFL::GreaterOp,
                                    ::mlir::mhlo::ComparisonDirection::GT>,
               ConvertMhloCompareOp<mlir::TFL::GreaterEqualOp,
                                    ::mlir::mhlo::ComparisonDirection::GE>>(
      patterns.getContext());

  ConversionTarget target(*context);
  // Intermediate dialects.
  target.addIllegalDialect<shape::ShapeDialect>();
  target.addIllegalDialect<::mlir::mhlo::MhloDialect>();
  // Final expected dialects.
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<TFL::TensorFlowLiteDialect>();

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  if (failed(applyPartialConversion(func, target, frozen_patterns))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFMhloTFLPass() {
  return std::make_unique<TFMhloTFLPass>();
}

static PassRegistration<TFMhloTFLPass> pass([] {
  return CreateTFMhloTFLPass();
});

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
