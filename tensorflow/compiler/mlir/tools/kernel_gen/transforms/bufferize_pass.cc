/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for translating mixed IR to buffer form.
// Currently it supports MHLO and some operations from the Standard dialect.

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for bufferization.

static Value materializeTensorLoad(OpBuilder& builder, TensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<TensorLoadOp>(loc, type, inputs[0]);
}

// TODO(pifon): Remove as soon as https://reviews.llvm.org/D93126 is landed.
class CustomBufferizeTypeConverter : public BufferizeTypeConverter {
 public:
  CustomBufferizeTypeConverter() {
    // Keep all types unchanged.
    addConversion([](Type type) { return type; });
    // Convert RankedTensorType to MemRefType.
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    // Convert UnrankedTensorType to UnrankedMemRefType.
    addConversion([](UnrankedTensorType type) -> Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    addArgumentMaterialization(materializeTensorLoad);
    addSourceMaterialization(materializeTensorLoad);
    addTargetMaterialization([](OpBuilder& builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memref_type = inputs[0].getType().dyn_cast<MemRefType>()) {
        assert(!memref_type.getAffineMaps().empty());
        return inputs[0];
      }
      assert(inputs[0].getType().isa<TensorType>());
      return builder.create<TensorToMemrefOp>(loc, type, inputs[0]);
    });
  }
};

struct HloBufferizePass : public HloBufferizePassBase<HloBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo::LmhloDialect>();
  }

 public:
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<complex::ComplexDialect, lmhlo::LmhloDialect,
                           StandardOpsDialect, tensor::TensorDialect,
                           math::MathDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();

    CustomBufferizeTypeConverter converter;
    // Configure bufferize pattern for functions and lhlo.
    mhlo::populateDynamicHLOToLHLOConversionPattern(
        &context, &converter, &patterns, /*insert_copy=*/false);
    populateFuncOpTypeConversionPattern(patterns, &context, converter);
    populateCallOpTypeConversionPattern(patterns, &context, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, &context,
                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, &context, converter);

    // Configure legality and structural patterns.
    populateBufferizeMaterializationLegality(target);
    linalg::populateLinalgBufferizePatterns(&context, converter, patterns);
    populateShapeStructuralTypeConversionsAndLegality(&context, converter,
                                                      patterns, target);
    scf::populateSCFStructuralTypeConversionsAndLegality(&context, converter,
                                                         patterns, target);
    // TODO(herhut): Move this legality configuration to bufferize itself?
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto inputs = op.getType().getInputs();
      auto results = op.getType().getResults();
      return converter.isLegal(inputs) && converter.isLegal(results) &&
             converter.isLegal(&op.getBody());
    });
    auto isLegalOp = [&](Operation* op) { return converter.isLegal(op); };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOp);
    target.addDynamicallyLegalOp<CallOp, ReturnOp>(isLegalOp);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct FinalBufferizePass : public FinalBufferizePassBase<FinalBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, scf::SCFDialect, shape::ShapeDialect,
                    tensor::TensorDialect, tf_framework::TFFrameworkDialect,
                    lmhlo::LmhloDialect>();
  }

 public:
  void runOnOperation() override {
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<complex::ComplexDialect, scf::SCFDialect,
                           StandardOpsDialect, tensor::TensorDialect,
                           tf_framework::TFFrameworkDialect, AffineDialect,
                           shape::ShapeDialect, lmhlo::LmhloDialect,
                           linalg::LinalgDialect, math::MathDialect>();
    target.addLegalOp<FuncOp, ModuleOp, ModuleTerminatorOp>();

    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::GenerateOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::CastOp,
                        chlo::MinimumBroadcastShapesOp, TensorLoadOp,
                        TensorToMemrefOp>();
    BufferizeTypeConverter converter;
    auto typesAreLegal = [&converter](Operation* op) {
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    };
    target.addDynamicallyLegalOp<ConstantOp, DimOp, RankOp, SelectOp>(
        typesAreLegal);

    OwningRewritePatternList patterns;
    populateTensorBufferizePatterns(&context, converter, patterns);
    populateStdBufferizePatterns(&context, converter, patterns);
    populateEliminateBufferizeMaterializationsPatterns(&context, converter,
                                                       patterns);
    populateExtraStdBufferizePattern(&context, &converter, &patterns);
    populateShapeStructuralTypeConversionsAndLegality(&context, converter,
                                                      patterns, target);
    scf::populateSCFStructuralTypeConversionsAndLegality(&context, converter,
                                                         patterns, target);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateHloBufferizePass() {
  return std::make_unique<HloBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp> > CreateFinalBufferizePass() {
  return std::make_unique<FinalBufferizePass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
