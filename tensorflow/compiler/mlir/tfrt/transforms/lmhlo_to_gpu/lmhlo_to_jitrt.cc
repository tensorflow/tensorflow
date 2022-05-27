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
#include <numeric>
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
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
using mlir::arith::ConstantOp;
using mlir::arith::IndexCastOp;
using mlir::detail::PassOptions;
using mlir::func::CallOp;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::gpu::GPUModuleOp;
using mlir::gpu::LaunchFuncOp;
using mlir::gpu::MemcpyOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;
using mlir::lmhlo::TerminatorOp;
using mlir::lmhlo::WhileOp;
using mlir::lmhlo_gpu::CholeskyOp;
using mlir::lmhlo_gpu::GEMM_BiasOp;
using mlir::lmhlo_gpu::GEMMOp;
using mlir::memref::GetGlobalOp;

class ConvertLmhloConstantToArgPass
    : public ConvertLmhloConstantToArgPassBase<ConvertLmhloConstantToArgPass> {
 public:
  ConvertLmhloConstantToArgPass() = default;
  explicit ConvertLmhloConstantToArgPass(int64_t min_num_elements) {
    this->min_num_elements_ = min_num_elements;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
  }
};

class ConvertGpuToJitRtPass
    : public ConvertGpuToJitRtPassBase<ConvertGpuToJitRtPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
  }
};

class ConvertLmhloGpuToJitRtPass
    : public ConvertLmhloGpuToJitRtPassBase<ConvertLmhloGpuToJitRtPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect,
                    mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();
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

template <typename IoFeedOp>
class IoFeedOpLowering : public OpRewritePattern<IoFeedOp> {
 public:
  explicit IoFeedOpLowering(MLIRContext* ctx)
      : OpRewritePattern<IoFeedOp>(ctx) {}

  static llvm::StringRef Name(InfeedOp) { return "infeed"; }
  static llvm::StringRef Name(OutfeedOp) { return "outfeed"; }

  LogicalResult matchAndRewrite(IoFeedOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Custom call target.
    NamedAttribute target(b.getStringAttr("rt.direct_custom_call"),
                          b.getStringAttr(Twine("xla.gpu.") + Name(op)));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), Name(op), custom_call_type,
                                      custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(op->template getParentOfType<ModuleOp>());
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());
    call->setAttr(b.getStringAttr("config"), op.configAttr());

    // Erase the original infeed/outfeed operation.
    rewriter.eraseOp(op);

    return success();
  }
};

class InfeedOpLowering : public IoFeedOpLowering<InfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

class OutfeedOpLowering : public IoFeedOpLowering<OutfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

// -------------------------------------------------------------------------- //

class MemcpyOpLowering : public OpRewritePattern<MemcpyOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  // We use a heuristic to identify the direction of the memcpy operation, if
  // the operand was allocated by alloca op or is a global memref, then it must
  // be a memref on the host.
  static bool IsHostMemRef(Value value) {
    auto* op = value.getDefiningOp();
    return llvm::isa_and_nonnull<memref::AllocaOp, memref::GetGlobalOp>(op);
  }

  LogicalResult matchAndRewrite(MemcpyOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Identify the direction of the memcpy operation.
    auto memcpy = [&]() {
      if (IsHostMemRef(op.dst())) return "memcpy.d2h";
      if (IsHostMemRef(op.src())) return "memcpy.h2d";
      return "memcpy.d2d";
    }();

    // Custom call target.
    NamedAttribute target(b.getStringAttr("rt.direct_custom_call"),
                          b.getStringAttr(Twine("xla.gpu.") + memcpy));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), memcpy, custom_call_type,
                                      custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(op->getParentOfType<ModuleOp>());
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Create a function launch call operation.
    rewriter.replaceOpWithNewOp<CallOp>(op, inserted, TypeRange(),
                                        op.getOperands());

    return success();
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
  GemmUidGenerator() : cnt_(0) {}
  int64_t uid() { return cnt_.fetch_add(1); }

 private:
  std::atomic<int64_t> cnt_;
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

class WhileOpLowering : public OpRewritePattern<WhileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Create an `scf.while` loop in place of `lmhlo.while` loop.
    auto loop = b.create<scf::WhileOp>(TypeRange(), ValueRange());

    // Predicate buffer placed on the device.
    assert(op.getNumOperands() == 1 && "expected single cond operand");
    Value pred = op.getOperand(0);

    // Clone condition and body blocks into the new loop operation.
    BlockAndValueMapping mapping;
    op.cond().cloneInto(&loop.getBefore(), mapping);
    op.body().cloneInto(&loop.getAfter(), mapping);

    {  // Replace loop condition terminator.
      auto* terminator = loop.getBefore().back().getTerminator();
      b.setInsertionPointAfter(terminator);

      // Copy predicate buffer to the host ...
      auto i1 = b.getI1Type();
      Value pred_on_host = b.create<memref::AllocaOp>(MemRefType::get({}, i1));
      b.create<gpu::MemcpyOp>(TypeRange(), ValueRange({pred_on_host, pred}));

      // .. and check if we need to continue loop iteration.
      Value cond = b.create<memref::LoadOp>(i1, pred_on_host, ValueRange());
      b.create<scf::ConditionOp>(cond, ValueRange());
      rewriter.eraseOp(terminator);
    }

    {  // Replace loop body terminator.
      auto* terminator = loop.getAfter().back().getTerminator();
      b.setInsertionPointAfter(terminator);
      b.create<scf::YieldOp>(TypeRange(), ValueRange());
      rewriter.eraseOp(terminator);
    }

    // Erase the original while loop.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

using GlobalConstantsArgs = llvm::DenseMap<FuncOp, llvm::StringMap<Value>>;

// Returns a mapping from a global constant name to the function argument.
//
// Example:
//
//   memref.global "private" constant @cst : memref<2x3xf32>
//   func @get_global(%arg0: memref<24xi8> {lmhlo.constant_name = "cst"})
//
// All memref.get_global operations will be replaced by constant arguments
// corresponding to the global constant.
GlobalConstantsArgs GetConstantArgs(ModuleOp m) {
  GlobalConstantsArgs mapping;

  m.walk([&](FuncOp func) {
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      auto cst = func.getArgAttrOfType<StringAttr>(i, "lmhlo.constant_name");
      if (cst) mapping[func][cst] = func.getArgument(i);
    }
  });

  return mapping;
}

class GetGlobalOpLowering : public OpRewritePattern<GetGlobalOp> {
 public:
  GetGlobalOpLowering(MLIRContext* ctx, const GlobalConstantsArgs& cst_args)
      : OpRewritePattern<GetGlobalOp>(ctx), cst_args_(cst_args) {}

  LogicalResult matchAndRewrite(GetGlobalOp op,
                                PatternRewriter& rewriter) const override {
    // Find global constants mapping for the parent function.
    auto func_mapping = cst_args_.find(op->getParentOfType<FuncOp>());
    if (func_mapping == cst_args_.end()) return failure();

    // Check if the global operation correposponds to the LMHLO constant arg.
    auto arg = func_mapping->second.find(op.name());
    if (arg == func_mapping->second.end()) return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefType memref = op->getResult(0).getType().cast<MemRefType>();

    // For identity layouts we can replace all loads from a global with the
    // corresponding argument.
    if (memref.getLayout().isIdentity()) {
      Value c0 = b.create<ConstantOp>(rewriter.getIndexAttr(0));
      rewriter.replaceOpWithNewOp<memref::ViewOp>(op, memref, arg->second, c0,
                                                  ValueRange());
      return success();
    }

    // For non-identity type we first view constant argument as a flat memref
    // with the correct element type, and then cast it to the strided memref
    // corresponding to the original memref layout.

    // Get the strides and offset from the original memref type.
    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (failed(getStridesAndOffset(memref, strides, offset)))
      return op.emitOpError("failed to compute strides and offset");

    // Create a 1d view into the corresponding argument.
    Value c0 = b.create<ConstantOp>(rewriter.getIndexAttr(0));
    Value flat_view = b.create<memref::ViewOp>(
        MemRefType::get({memref.getNumElements()}, memref.getElementType()),
        arg->second, c0, ValueRange());

    // Cast flat memref view into the original memref type.
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, memref, flat_view, offset, memref.getShape(), strides);

    return success();
  }

 private:
  const GlobalConstantsArgs& cst_args_;
};

// -------------------------------------------------------------------------- //

class CholeskyOpLowering : public OpRewritePattern<CholeskyOp> {
 public:
  explicit CholeskyOpLowering(MLIRContext* ctx)
      : OpRewritePattern<CholeskyOp>(ctx) {}

  LogicalResult matchAndRewrite(CholeskyOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr("rt.direct_custom_call"),
                          b.getStringAttr("xla.gpu.cholesky"));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), "cholesky", custom_call_type,
                                      custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert Cholesky to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    const auto& dims = op.input().getType().cast<mlir::MemRefType>().getShape();
    if (dims.size() < 2)
      return op.emitOpError() << "Input's dimension count (" << dims.size()
                              << ") must be 2 or greater.";
    int64_t n = dims[dims.size() - 1];
    int64_t batch_size =
        std::accumulate(dims.begin(), dims.end() - 2, int64_t{1},
                        [](int64_t a, int64_t b) { return a * b; });

    // Copy backend specific attributes.
    call->setAttr(b.getStringAttr("batch_size"),
                  b.getI64IntegerAttr(batch_size));
    call->setAttr(b.getStringAttr("n"), b.getI64IntegerAttr(n));
    call->setAttr(b.getStringAttr("uplo"), b.getI64IntegerAttr(op.is_lower()));

    // Erase the original Cholesky operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

void ConvertLmhloConstantToArgPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Replace memref loads from globals corresponding to the constant arguments.
  RewritePatternSet patterns(ctx);
  GlobalConstantsArgs cst_args = GetConstantArgs(module);
  patterns.insert<GetGlobalOpLowering>(ctx, cst_args);

  // Set up conversion target to rewrite only GetGlobalOp larger than the
  // threshold and avoid any other canonicalizations that can break later
  // passes.
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<GetGlobalOp>([&](GetGlobalOp op) {
    auto memref = op.getType();
    return memref.getNumElements() < min_num_elements_;
  });
  target.addLegalOp<ConstantOp, memref::ViewOp, memref::ReinterpretCastOp>();

  // TODO(ezhulenev): By adding MHLO and LMHLO to a set of legal dialects, we
  // suppress any rewrites for these dialects (there are canonicalization
  // patterns that interact badly with downstream Gpu binary code generation).
  target.addLegalDialect<mhlo::MhloDialect, lmhlo::LmhloDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void ConvertGpuToJitRtPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Convert gpu operations to JitRt gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<GpuModuleOpLowering, LaunchFuncOpLowering, MemcpyOpLowering,
                  InfeedOpLowering, OutfeedOpLowering>(ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

void ConvertLmhloGpuToJitRtPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  GemmUidGenerator uid;

  // Convert lmhlo_gpu operations to JitRt gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<GemmOpLowering, GemmBiasOpLowering>(ctx, uid);
  patterns.insert<CholeskyOpLowering, WhileOpLowering, TerminatorOpLowering>(
      ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertGpuToJitRtPass() {
  return std::make_unique<ConvertGpuToJitRtPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertLmhloConstantToArgPass() {
  return std::make_unique<ConvertLmhloConstantToArgPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertLmhloConstantToArgPass(
    int64_t min_num_elements) {
  return std::make_unique<ConvertLmhloConstantToArgPass>(min_num_elements);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertLmhloGpuToJitRtPass() {
  return std::make_unique<ConvertLmhloGpuToJitRtPass>();
}

void populateLmhloToJitRtPasses(mlir::OpPassManager& pm) {
  // Convert large global memrefs corresponding to XLA constants with arguments,
  // so that compiled device kernels do not capture them.
  //
  // TODO(ezhulenev): Threshold should be consistent with the device kernel
  // code generation. If constant will be embedded into the device module, we
  // should not inline it too early. Currently it's hardcoded to `1` element.
  pm.addPass(createConvertLmhloConstantToArgPass(/*min_num_elements=*/2));

  // Small global constants will be embedded into the device modules.
  pm.addPass(createConvertLmhloToGpuBinaryPass());

  // Convert remaining small global memrefs corresponding to constant arguments.
  pm.addPass(createConvertLmhloConstantToArgPass());

  // Lower all Gpu operations to the JitRt Gpu runtime intrinsics.
  pm.addPass(createConvertLmhloGpuToJitRtPass());
  pm.addPass(createConvertGpuToJitRtPass());
}

void registerLmhloToJitRtPasses() {
  mlir::registerPass([] { return createConvertGpuToJitRtPass(); });
  mlir::registerPass([] { return createConvertLmhloConstantToArgPass(); });
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
