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

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/bufferizable_op_interface_impl.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
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

static Value materializeToTensor(OpBuilder& builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

// TODO(pifon): Remove as soon as https://reviews.llvm.org/D93126 is landed.
class CustomBufferizeTypeConverter
    : public bufferization::BufferizeTypeConverter {
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
    addArgumentMaterialization(materializeToTensor);
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization([](OpBuilder& builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memref_type = inputs[0].getType().dyn_cast<MemRefType>()) {
        assert(!memref_type.getLayout().isIdentity());
        return inputs[0];
      }
      assert(inputs[0].getType().isa<TensorType>());
      return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
    });
  }
};

struct ComputeOpAndFuncBufferizePass
    : public ComputeOpAndFuncBufferizePassBase<ComputeOpAndFuncBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<linalg::LinalgDialect, bufferization::BufferizationDialect,
                    lmhlo::LmhloDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, vector::VectorDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mhlo::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<linalg::LinalgDialect, mhlo::MhloDialect,
                                 shape::ShapeDialect, tensor::TensorDialect,
                                 vector::VectorDialect>();
    // Ops inside TiledLoopOps have special handling.
    options.denyOperationInFilter([](Operation* op) {
      return mlir::isa<gml_st::LoopOp>(op->getParentOp());
    });
    // Configure bufferization options for mhlo ops.
    options.addDialectStateInitializer(
        mhlo::MhloDialect::getDialectNamespace(), []() {
          auto dialect_state = std::make_unique<mhlo::MhloBufferizationState>();
          dialect_state->enforce_identity_map_fn = [](Operation* op) {
            // Force identity maps for several ops which don't support memrefs
            // with affine_maps.
            return llvm::any_of(op->getUsers(), [](Operation* user) {
              return isa<gml_st::LoopOp, func::ReturnOp, mhlo::DynamicReshapeOp,
                         tensor::CastOp, tensor::CollapseShapeOp,
                         tensor::ExpandShapeOp>(user);
            });
          };
          return dialect_state;
        });

    if (failed(bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
    RewritePatternSet patterns(&getContext());
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, complex::ComplexDialect, lmhlo::LmhloDialect,
        AffineDialect, vector::VectorDialect, memref::MemRefDialect,
        func::FuncDialect, tensor::TensorDialect, math::MathDialect>();
    target.addLegalOp<UnrealizedConversionCastOp, gml_st::LoopOp>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addDynamicallyLegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp>(
        [&](Operation* op) {
          return mlir::isa<gml_st::LoopOp>(op->getParentOp());
        });

    CustomBufferizeTypeConverter converter;
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    // Configure legality and structural patterns.
    bufferization::populateBufferizeMaterializationLegality(target);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    // TODO(herhut): Move this legality configuration to bufferize itself?
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto inputs = op.getType().getInputs();
      auto results = op.getType().getResults();
      return converter.isLegal(inputs) && converter.isLegal(results) &&
             converter.isLegal(&op.getBody());
    });
    auto isLegalOp = [&](Operation* op) { return converter.isLegal(op); };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(isLegalOp);

    auto isLegalOrInsideTiledLoop = [&](Operation* op) {
      return converter.isLegal(op) ||
             mlir::isa<gml_st::LoopOp>(op->getParentOp());
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        isLegalOrInsideTiledLoop);
    target
        .addDynamicallyLegalOp<vector::TransferWriteOp, vector::TransferReadOp>(
            isLegalOrInsideTiledLoop);

    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

struct TiledLoopBufferizePass
    : public TiledLoopBufferizePassBase<TiledLoopBufferizePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    RewritePatternSet patterns(&getContext());
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<shape::ShapeDialect>();
    if (failed(bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
    RewritePatternSet patterns(&getContext());
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        complex::ComplexDialect, lmhlo::LmhloDialect, AffineDialect,
        vector::VectorDialect, memref::MemRefDialect, func::FuncDialect,
        tensor::TensorDialect, math::MathDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp>();

    CustomBufferizeTypeConverter converter;
    mhlo::RemoveSignTypeConverter remove_sign_converter;

    // Configure bufferize pattern.
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    bufferization::populateBufferizeMaterializationLegality(target);
    populateTiledLoopBufferizePattern(&getContext(), &converter, &patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    // Configure legality.
    auto isLegalOp = [&](Operation* op) { return converter.isLegal(op); };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOp);
    target.addDynamicallyLegalOp<func::CallOp, gml_st::LoopOp, gml_st::YieldOp,
                                 LLVM::InlineAsmOp, vector::TransferWriteOp,
                                 vector::TransferReadOp>(isLegalOp);

    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

struct FinalBufferizePass : public FinalBufferizePassBase<FinalBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, shape::ShapeDialect, tensor::TensorDialect,
                    tf_framework::TFFrameworkDialect, lmhlo::LmhloDialect,
                    arith::ArithmeticDialect, vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }
  // Default alignment_ specified in passes.td
  FinalBufferizePass() = default;

  explicit FinalBufferizePass(uint64_t alignment) { alignment_ = alignment; }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    options.bufferAlignment = alignment_;
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<
        arith::ArithmeticDialect, linalg::LinalgDialect, func::FuncDialect,
        shape::ShapeDialect, tensor::TensorDialect, vector::VectorDialect>();
    if (failed(bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        cf::ControlFlowDialect, complex::ComplexDialect, memref::MemRefDialect,
        func::FuncDialect, scf::SCFDialect, tensor::TensorDialect,
        tf_framework::TFFrameworkDialect, AffineDialect, shape::ShapeDialect,
        lmhlo::LmhloDialect, linalg::LinalgDialect, math::MathDialect,
        vector::VectorDialect>();
    target.addLegalOp<FuncOp, ModuleOp>();

    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::GenerateOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::CastOp, tensor::DimOp,
                        tensor::RankOp, chlo::MinimumBroadcastShapesOp,
                        bufferization::ToTensorOp, bufferization::ToMemrefOp,
                        tensor::ExpandShapeOp, tensor::CollapseShapeOp>();
    bufferization::BufferizeTypeConverter converter;
    auto typesAreLegal = [&converter](Operation* op) {
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    };
    target.addDynamicallyLegalOp<func::ConstantOp, arith::ConstantOp,
                                 arith::IndexCastOp, arith::SelectOp,
                                 gml_st::LoopOp, gml_st::YieldOp,
                                 tf_framework::JITExecuteOp>(typesAreLegal);

    RewritePatternSet patterns(&getContext());
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);
    populateExtraBufferizePatterns(&getContext(), &converter, &patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateComputeOpAndFuncBufferizePass() {
  return std::make_unique<ComputeOpAndFuncBufferizePass>();
}

std::unique_ptr<OperationPass<FuncOp>> CreateTiledLoopBufferizePass() {
  return std::make_unique<TiledLoopBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass() {
  return std::make_unique<FinalBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass(
    uint64_t alignment) {
  return std::make_unique<FinalBufferizePass>(alignment);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
