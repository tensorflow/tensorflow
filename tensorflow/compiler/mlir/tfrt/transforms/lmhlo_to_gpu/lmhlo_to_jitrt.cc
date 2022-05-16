// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_jitrt.h"

#include <cstdint>
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu_binary.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/jitrt_passes.h.inc"

using lmhlo_gpu::GEMM_BiasOp;
using mlir::DialectRegistry;
using mlir::FunctionType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::OperationPass;
using mlir::success;
using mlir::SymbolTable;
using mlir::Type;
using mlir::TypeRange;
using mlir::arith::IndexCastOp;
using mlir::detail::PassOptions;
using mlir::func::CallOp;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::gpu::GPUModuleOp;
using mlir::gpu::LaunchFuncOp;
using mlir::lmhlo::TerminatorOp;
using mlir::lmhlo_gpu::GEMMOp;

class ConvertGpuBinaryToJitRtPass
    : public ConvertGpuBinaryToJitRtPassBase<ConvertGpuBinaryToJitRtPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
  }
};

class ConvertLmhloGpuToJitRtPass
    : public ConvertLmhloGpuToJitRtPassBase<ConvertLmhloGpuToJitRtPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
  }
};

}  // namespace

// -------------------------------------------------------------------------- //

class GpuModuleOpLowering : public OpRewritePattern<GPUModuleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GPUModuleOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// -------------------------------------------------------------------------- //

class TerminatorOpLowering : public OpRewritePattern<TerminatorOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TerminatorOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return mlir::success();
  }
};

// -------------------------------------------------------------------------- //

class LaunchFuncOpLowering : public OpRewritePattern<LaunchFuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LaunchFuncOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Cast grid and block dimensions to i32 before passing to the custom call.
    auto cast = [&](mlir::Value value) {
      return b.create<IndexCastOp>(b.getI32Type(), value);
    };

    // Prepare arguments for the custom call.
    llvm::SmallVector<Value> args = {
        cast(op.gridSizeX()),  cast(op.gridSizeY()),  cast(op.gridSizeZ()),
        cast(op.blockSizeX()), cast(op.blockSizeY()), cast(op.blockSizeZ())};

    // Add kernel arguments.
    llvm::copy(op.operands(), std::back_inserter(args));

    // Types of the custom call arguments.
    llvm::SmallVector<Type> args_types = TypeRange(ValueRange(args));

    // Custom call target.
    NamedAttribute target(b.getStringAttr("rt.direct_custom_call"),
                          b.getStringAttr("xla.gpu.func.launch"));

    // Create a custom call function declaration.
    auto custom_call_type = FunctionType::get(ctx, args_types, TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), "launch_func",
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Get the compiled gpu function.
    auto* kernel = SymbolTable::lookupNearestSymbolFrom(op, op.kernel());
    assert(kernel && "kernel not found");

    // Get the compiled GPU binary from the device kernel module.
    auto gpu_module = kernel->getParentOfType<mlir::gpu::GPUModuleOp>();
    auto gpu_binary = gpu_module->getAttrOfType<mlir::StringAttr>("binary");

    // Create a function launch call operation.
    auto call =
        rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(), args);
    call->setAttr(b.getStringAttr("ptx"), gpu_binary);
    call->setAttr(b.getStringAttr("kernel"), op.getKernelName());

    // Erase the original gpu launch operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

// Every Gemm operation in the module gets assigned a unique id, that is passed
// to the custom call handler. This id is used for caching resources between the
// different invocations of the same gemm operation.
class GemmUidGenerator {
 public:
  int64_t uid() { return cnt_.fetch_add(1); }

 private:
  std::atomic<int64_t> cnt_ = 0;
};

template <typename Gemm>
class GemmLowering : public OpRewritePattern<Gemm> {
 private:
  static StringRef CustomCallTarget(GEMMOp) { return "xla.gpu.gemm"; }
  static StringRef CustomCallTarget(GEMM_BiasOp) { return "xla.gpu.gemm.bias"; }

  static void SetOptionalAttrs(ImplicitLocOpBuilder& b, GEMMOp op,
                               CallOp call) {}
  static void SetOptionalAttrs(ImplicitLocOpBuilder& b, GEMM_BiasOp op,
                               CallOp call) {
    call->setAttr(b.getStringAttr("beta"), op.betaAttr());
  }

 public:
  GemmLowering(MLIRContext* ctx, GemmUidGenerator& uid)
      : OpRewritePattern<Gemm>(ctx), uid_(uid) {}

  LogicalResult matchAndRewrite(Gemm op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->template getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr("rt.direct_custom_call"),
                          b.getStringAttr(CustomCallTarget(op)));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), "gemm", custom_call_type,
                                      custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert Gemm to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    // Assign a unique id to this instance of a gemm operation.
    call->setAttr(b.getStringAttr("uid"), b.getI64IntegerAttr(uid_.uid()));

    // Copy backend specific attributes.
    call->setAttr(b.getStringAttr("algorithm"), op.algorithmAttr());
    call->setAttr(b.getStringAttr("alpha_imag"), op.alpha_imagAttr());
    call->setAttr(b.getStringAttr("alpha_real"), op.alpha_realAttr());

    // Set optional arguments that are defined only for some Gemm ops.
    SetOptionalAttrs(b, op, call);

    // TODO(ezhulenev): Once cutom calls support passing structured attributes
    // we should be able to pass `mhlo.dot` attribute directly.
    auto dot = op.dot_dimension_numbers();
    auto lhs_batch = b.getI64TensorAttr(dot.getLhsBatchingDimensions());
    auto lhs_contract = b.getI64TensorAttr(dot.getLhsContractingDimensions());
    auto rhs_batch = b.getI64TensorAttr(dot.getRhsBatchingDimensions());
    auto rhs_contract = b.getI64TensorAttr(dot.getRhsContractingDimensions());

    call->setAttr(b.getStringAttr("lhs_batching_dimensions"), lhs_batch);
    call->setAttr(b.getStringAttr("lhs_contracting_dimensions"), lhs_contract);
    call->setAttr(b.getStringAttr("rhs_batching_dimensions"), rhs_batch);
    call->setAttr(b.getStringAttr("rhs_contracting_dimensions"), rhs_contract);

    // Erase the original gemm operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  GemmUidGenerator& uid_;
};

class GemmOpLowering : public GemmLowering<GEMMOp> {
 public:
  using GemmLowering::GemmLowering;
};

class GemmBiasOpLowering : public GemmLowering<GEMM_BiasOp> {
 public:
  using GemmLowering::GemmLowering;
};

// -------------------------------------------------------------------------- //

void ConvertGpuBinaryToJitRtPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Convert gpu operations to JitRt gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<GpuModuleOpLowering, LaunchFuncOpLowering>(ctx);

  // Set up conversion target to rewrite gpu operations.
  ConversionTarget target(*ctx);
  target.addIllegalOp<GPUModuleOp, LaunchFuncOp>();
  target.addLegalOp<IndexCastOp, FuncOp, CallOp, ReturnOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void ConvertLmhloGpuToJitRtPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  GemmUidGenerator uid;

  // Convert lmhlo_gpu operations to JitRt gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<GemmOpLowering, GemmBiasOpLowering>(ctx, uid);
  patterns.insert<TerminatorOpLowering>(ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertGpuBinaryToJitRtPass() {
  return std::make_unique<ConvertGpuBinaryToJitRtPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertLmhloGpuToJitRtPass() {
  return std::make_unique<ConvertLmhloGpuToJitRtPass>();
}

void populateLmhloToJitRtPasses(mlir::OpPassManager& pm) {
  pm.addPass(createConvertLmhloToGpuBinaryPass());
  pm.addPass(createConvertGpuBinaryToJitRtPass());
  pm.addPass(createConvertLmhloGpuToJitRtPass());
}

void registerLmhloToJitRtPasses() {
  mlir::registerPass([] { return createConvertGpuBinaryToJitRtPass(); });
  mlir::registerPass([] { return createConvertLmhloGpuToJitRtPass(); });

  mlir::registerPassPipeline(
      "lmhlo-to-jitrt", "Lower LMHLO to JitRt IR",
      [](OpPassManager& pm, StringRef options,
         function_ref<LogicalResult(const Twine&)> errorHandler) {
        populateLmhloToJitRtPasses(pm);
        return success();
      },
      /*optHandler=*/[](function_ref<void(const PassOptions&)>) {});
}

}  // namespace tensorflow
