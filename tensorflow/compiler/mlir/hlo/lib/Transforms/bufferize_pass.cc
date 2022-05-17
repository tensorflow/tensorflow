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
#include <functional>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/bufferizable_op_interface_impl.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/bufferizable_op_interface_impl.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir-hlo/Transforms/rewriters.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

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
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    // Configure legality and structural patterns.
    bufferization::populateBufferizeMaterializationLegality(target);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    // TODO(herhut): Move this legality configuration to bufferize itself?
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto inputs = op.getFunctionType().getInputs();
      auto results = op.getFunctionType().getResults();
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

struct OneShotBufferizePass
    : public OneShotBufferizeBase<OneShotBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<linalg::LinalgDialect, bufferization::BufferizationDialect,
                    lmhlo::LmhloDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, mhlo::MhloDialect, scf::SCFDialect,
                    shape::ShapeDialect, vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    gml_st::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mhlo::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions opts;
    opts.allowReturnAllocs = true;
    opts.dropEquivalentFuncResults = false;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion =
        bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
    opts.createDeallocs = false;
    opts.bufferAlignment = 64;

    ModuleOp module = getOperation();
    if (failed(bufferization::runOneShotModuleBufferize(module, opts))) {
      signalPassFailure();
    }
  }
};

struct FinalBufferizePass : public FinalBufferizePassBase<FinalBufferizePass> {
 private:
  BufferizeDialectsCallback dialects_callback;
  BufferizePatternsCallback patterns_callback;

 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, shape::ShapeDialect, tensor::TensorDialect,
                    lmhlo::LmhloDialect, arith::ArithmeticDialect,
                    vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
    if (dialects_callback) dialects_callback(registry);
  }
  // Default alignment_ specified in passes.td
  FinalBufferizePass() = default;

  explicit FinalBufferizePass(uint64_t alignment) { alignment_ = alignment; }

  void setCallbacks(BufferizeDialectsCallback dc,
                    BufferizePatternsCallback pc) {
    dialects_callback = dc;
    patterns_callback = pc;
  }

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
        AffineDialect, shape::ShapeDialect, lmhlo::LmhloDialect,
        linalg::LinalgDialect, math::MathDialect, vector::VectorDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp>();

    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::GenerateOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::CastOp, tensor::DimOp,
                        tensor::RankOp, chlo::MinimumBroadcastShapesOp,
                        bufferization::ToTensorOp, bufferization::ToMemrefOp,
                        tensor::ExpandShapeOp, tensor::CollapseShapeOp>();
    CustomBufferizeTypeConverter converter;
    auto typesAreLegal = [&converter](Operation* op) {
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    };
    target.addDynamicallyLegalOp<func::ConstantOp, arith::ConstantOp,
                                 arith::IndexCastOp, arith::SelectOp,
                                 gml_st::LoopOp, gml_st::YieldOp>(
        typesAreLegal);

    RewritePatternSet patterns(&getContext());
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);
    populateExtraBufferizePatterns(&getContext(), &converter, &patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (patterns_callback)
      patterns_callback(target, &getContext(), &converter, &patterns);

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }
};

}  // namespace

namespace hlo {
std::unique_ptr<OperationPass<ModuleOp>> CreateOneShotBufferizePass() {
  return std::make_unique<OneShotBufferizePass>();
}
}  // namespace hlo

std::unique_ptr<OperationPass<ModuleOp>> CreateComputeOpAndFuncBufferizePass() {
  return std::make_unique<ComputeOpAndFuncBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass() {
  return std::make_unique<FinalBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass(
    uint64_t alignment, BufferizeDialectsCallback dc,
    BufferizePatternsCallback pc) {
  auto pass = std::make_unique<FinalBufferizePass>(alignment);
  pass->setCallbacks(dc, pc);
  return pass;
}

}  // namespace mlir
