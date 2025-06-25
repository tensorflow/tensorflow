/* Copyright 2020 The OpenXLA Authors.

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
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/interfaces/bufferizable_op_interface_impl.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "transforms/passes.h"
#include "transforms/rewriters.h"

namespace mlir {

#define GEN_PASS_DEF_COMPUTEOPANDFUNCBUFFERIZEPASS
#define GEN_PASS_DEF_FINALBUFFERIZEPASS
#define GEN_PASS_DEF_ONESHOTBUFFERIZE
#include "transforms/passes.h.inc"

namespace {

// Label for functions created by fusion outlining.
static constexpr char kFusionFunctionLabel[] = "fusion";

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for bufferization.

static Value materializeToTensor(OpBuilder& builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(mlir::isa<BaseMemRefType>(inputs[0].getType()));
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

// TODO(pifon): Remove as soon as https://reviews.llvm.org/D93126 is landed.
class CustomBufferizeTypeConverter : public mlir::TypeConverter {
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
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization([](OpBuilder& builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1 && "expected exactly one input");
      if (auto inputType = dyn_cast<MemRefType>(inputs[0].getType())) {
        // MemRef to MemRef cast.
        assert(inputType != type && "expected different types");
        // Ranked to unranked casts must be explicit.
        auto rankedDestType = dyn_cast<MemRefType>(type);
        if (!rankedDestType) return nullptr;
        bufferization::BufferizationOptions options;
        options.bufferAlignment = 0;
        FailureOr<Value> replacement = castOrReallocMemRefValue(
            builder, inputs[0], rankedDestType, options);
        if (failed(replacement)) return nullptr;
        return *replacement;
      }
      if (isa<TensorType>(inputs[0].getType())) {
        // Tensor to MemRef cast.
        return builder.create<bufferization::ToBufferOp>(loc, type, inputs[0]);
      }
      llvm_unreachable("only tensor/memref input types supported");
    });
    addTargetMaterialization([](OpBuilder& builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memrefType = mlir::dyn_cast<MemRefType>(inputs[0].getType())) {
        assert(!memrefType.getLayout().isIdentity());
        return inputs[0];
      }
      assert(mlir::isa<TensorType>(inputs[0].getType()));
      return builder.create<bufferization::ToBufferOp>(loc, type, inputs[0]);
    });
  }
};

static bufferization::BufferizationOptions getPartialBufferizationOptions() {
  bufferization::BufferizationOptions options;
  options.allowUnknownOps = true;
  options.copyBeforeWrite = true;
  options.unknownTypeConverterFn =
      [](TensorType type, Attribute memorySpace,
         const bufferization::BufferizationOptions& options) {
        return bufferization::getMemRefTypeWithStaticIdentityLayout(
            type, memorySpace);
      };
  options.opFilter.allowDialect<bufferization::BufferizationDialect>();
  return options;
}

struct ComputeOpAndFuncBufferizePass
    : public impl::ComputeOpAndFuncBufferizePassBase<
          ComputeOpAndFuncBufferizePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mhlo::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    bufferization::BufferizationOptions options =
        getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.opFilter.allowDialect<bufferization::BufferizationDialect,
                                  linalg::LinalgDialect, mhlo::MhloDialect,
                                  shape::ShapeDialect, vector::VectorDialect>();
    bufferization::BufferizationState bufferizationState;
    if (failed(bufferization::bufferizeOp(getOperation(), options,
                                          bufferizationState))) {
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
    target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                           complex::ComplexDialect, func::FuncDialect,
                           math::MathDialect, memref::MemRefDialect,
                           tensor::TensorDialect, vector::VectorDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    auto isLegalMhloOp = [&](Operation* op) {
      return isa<mhlo::MinimumBroadcastShapesOp>(op);
    };
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>(isLegalMhloOp);

    CustomBufferizeTypeConverter converter;
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    // Configure legality and structural patterns.
    target.addLegalOp<bufferization::ToTensorOp, bufferization::ToBufferOp>();
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

    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOp);
    target
        .addDynamicallyLegalOp<vector::TransferWriteOp, vector::TransferReadOp>(
            isLegalOp);

    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

struct OneShotBufferizePass
    : public impl::OneShotBufferizeBase<OneShotBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, mhlo::MhloDialect, scf::SCFDialect,
                    shape::ShapeDialect, vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mhlo::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions opts;
    opts.allowReturnAllocsFromLoops = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionArgTypeConverterFn =
        [=](TensorType tensorType, Attribute memorySpace,
            FunctionOpInterface funcOp,
            const bufferization::BufferizationOptions& /*options*/) {
          // Functions created by fusion outlining should have fully dynamic
          // layout. All other functions (for now only "main") gets static
          // layout.
          if (funcOp->hasAttr(kFusionFunctionLabel))
            return bufferization::getMemRefTypeWithFullyDynamicLayout(
                tensorType, memorySpace);
          return bufferization::getMemRefTypeWithStaticIdentityLayout(
              tensorType, memorySpace);
        };
    opts.inferFunctionResultLayout = false;
    opts.bufferAlignment = 64;

    ModuleOp module = getOperation();
    bufferization::BufferizationState bufferizationState;
    if (failed(bufferization::runOneShotModuleBufferize(module, opts,
                                                        bufferizationState))) {
      signalPassFailure();
    }
  }
};

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeToTensorOp
    : public OpConversionPattern<bufferization::ToTensorOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      bufferization::ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getBuffer());
    return success();
  }
};

// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeToMemrefOp
    : public OpConversionPattern<bufferization::ToBufferOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      bufferization::ToBufferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getTensor());
    return success();
  }
};

}  // namespace

struct FinalBufferizePass
    : public impl::FinalBufferizePassBase<FinalBufferizePass> {
 private:
  BufferizeDialectsCallback dialectsCallback;
  BufferizePatternsCallback patternsCallback;

 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, shape::ShapeDialect, tensor::TensorDialect,
                    arith::ArithDialect, vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
    if (dialectsCallback) dialectsCallback(registry);
  }
  // Default alignment_ specified in passes.td
  FinalBufferizePass() = default;

  explicit FinalBufferizePass(uint64_t alignment) { alignment_ = alignment; }

  void setCallbacks(BufferizeDialectsCallback dc,
                    BufferizePatternsCallback pc) {
    dialectsCallback = std::move(dc);
    patternsCallback = std::move(pc);
  }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    bufferization::BufferizationOptions options =
        getPartialBufferizationOptions();
    options.bufferAlignment = alignment_;
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.opFilter.allowDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        linalg::LinalgDialect, func::FuncDialect, shape::ShapeDialect,
        tensor::TensorDialect, vector::VectorDialect>();
    bufferization::BufferizationState bufferizationState;
    if (failed(bufferization::bufferizeOp(getOperation(), options,
                                          bufferizationState))) {
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
        arith::ArithDialect, bufferization::BufferizationDialect,
        cf::ControlFlowDialect, complex::ComplexDialect, memref::MemRefDialect,
        func::FuncDialect, scf::SCFDialect, tensor::TensorDialect,
        affine::AffineDialect, shape::ShapeDialect, linalg::LinalgDialect,
        math::MathDialect, vector::VectorDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp>();

    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::GenerateOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::CastOp, tensor::DimOp,
                        tensor::RankOp, mhlo::MinimumBroadcastShapesOp,
                        bufferization::ToTensorOp, bufferization::ToBufferOp,
                        tensor::ExpandShapeOp, tensor::CollapseShapeOp>();
    CustomBufferizeTypeConverter converter;
    auto typesAreLegal = [&converter](Operation* op) {
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    };
    target.addDynamicallyLegalOp<func::ConstantOp, arith::ConstantOp,
                                 arith::IndexCastOp, arith::SelectOp>(
        typesAreLegal);

    RewritePatternSet patterns(&getContext());
    patterns.add<BufferizeToTensorOp, BufferizeToMemrefOp>(converter,
                                                           &getContext());
    populateExtraBufferizePatterns(&getContext(), &converter, &patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (patternsCallback)
      patternsCallback(target, &getContext(), &converter, &patterns);

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }
};

}  // namespace

namespace hlo {
std::unique_ptr<OperationPass<ModuleOp>> createOneShotBufferizePass() {
  return std::make_unique<OneShotBufferizePass>();
}
}  // namespace hlo

std::unique_ptr<OperationPass<ModuleOp>> createComputeOpAndFuncBufferizePass() {
  return std::make_unique<ComputeOpAndFuncBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createFinalBufferizePass() {
  return std::make_unique<FinalBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createFinalBufferizePass(
    uint64_t alignment, BufferizeDialectsCallback dc,
    BufferizePatternsCallback pc) {
  auto pass = std::make_unique<FinalBufferizePass>(alignment);
  pass->setCallbacks(std::move(dc), std::move(pc));
  return pass;
}

}  // namespace mlir
