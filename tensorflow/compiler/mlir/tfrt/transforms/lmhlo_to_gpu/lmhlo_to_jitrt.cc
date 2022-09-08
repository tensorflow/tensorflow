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
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/jitrt_passes.h.inc"

using mlir::Attribute;
using mlir::DialectRegistry;
using mlir::FunctionType;
using mlir::IntegerAttr;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::OperationPass;
using mlir::success;
using mlir::SymbolTable;
using mlir::Type;
using mlir::TypeRange;
using mlir::WalkResult;
using mlir::arith::ConstantOp;
using mlir::arith::IndexCastOp;
using mlir::detail::PassOptions;
using mlir::func::CallOp;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::gpu::GPUModuleOp;
using mlir::gpu::LaunchFuncOp;
using mlir::gpu::MemcpyOp;
using mlir::gpu::MemsetOp;
using mlir::lmhlo::AllGatherOp;
using mlir::lmhlo::AllReduceOp;
using mlir::lmhlo::AllToAllOp;
using mlir::lmhlo::CaseOp;
using mlir::lmhlo::CollectivePermuteOp;
using mlir::lmhlo::CustomCallOp;
using mlir::lmhlo::FftOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;
using mlir::lmhlo::PartitionIdOp;
using mlir::lmhlo::ReduceScatterOp;
using mlir::lmhlo::ReplicaIdOp;
using mlir::lmhlo::TerminatorOp;
using mlir::lmhlo::WhileOp;
using mlir::lmhlo_gpu::AllReduceDoneOp;
using mlir::lmhlo_gpu::AllReduceStartOp;
using mlir::lmhlo_gpu::CholeskyOp;
using mlir::lmhlo_gpu::ConvBackwardFilterOp;
using mlir::lmhlo_gpu::ConvBackwardInputOp;
using mlir::lmhlo_gpu::ConvForwardFusedOp;
using mlir::lmhlo_gpu::ConvForwardFusedSideInputOp;
using mlir::lmhlo_gpu::ConvForwardOp;
using mlir::lmhlo_gpu::CublasLtMatmulOp;
using mlir::lmhlo_gpu::GEMMOp;
using mlir::memref::AllocaOp;
using mlir::memref::GetGlobalOp;

static constexpr const char kDirectCustomCall[] = "rt.direct_custom_call";

class ConvertLmhloGpuToJitRtPass
    : public ConvertLmhloGpuToJitRtPassBase<ConvertLmhloGpuToJitRtPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect,
                    mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                    mlir::cf::ControlFlowDialect>();
  }
};

}  // namespace

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

class GemmOpLowering : public OpRewritePattern<GEMMOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.gemm";

 public:
  GemmOpLowering(MLIRContext* ctx, GemmUidGenerator& uid)
      : OpRewritePattern<GEMMOp>(ctx), uid_(uid) {}

  LogicalResult matchAndRewrite(GEMMOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->template getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(kCustomCallTarget));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), kCustomCallTarget,
                                      custom_call_type, custom_call_attrs);
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
    auto algorithm_attr = op.getAlgorithm()
                              ? op.getAlgorithmAttr()
                              : b.getI64IntegerAttr(se::blas::kDefaultGemmAlgo);
    call->setAttr(b.getStringAttr("algorithm"), algorithm_attr);
    call->setAttr(b.getStringAttr("alpha_imag"), op.getAlphaImagAttr());
    call->setAttr(b.getStringAttr("alpha_real"), op.getAlphaRealAttr());
    call->setAttr(b.getStringAttr("beta"), op.getBetaAttr());
    call->setAttr(b.getStringAttr("dot_dims"), op.getDotDimensionNumbers());

    // Erase the original gemm operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  GemmUidGenerator& uid_;
};

// -------------------------------------------------------------------------- //

class CublasLtMatmulOpLowering : public OpRewritePattern<CublasLtMatmulOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.cublas.lt.matmul";

 public:
  CublasLtMatmulOpLowering(MLIRContext* ctx, GemmUidGenerator& uid)
      : OpRewritePattern<CublasLtMatmulOp>(ctx), uid_(uid) {}

  LogicalResult matchAndRewrite(CublasLtMatmulOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->template getParentOfType<ModuleOp>();

    std::string matmul;
    switch (op.getOperands().size()) {
      case 4:
        matmul = kCustomCallTarget;
        break;
      case 5:
        matmul = absl::StrCat(kCustomCallTarget, ".bias");
        break;
      default:
        return op.emitOpError("unexpected number of operands for matmul");
    }

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(matmul));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), matmul, custom_call_type,
                                      custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert matmul to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    // Assign a unique id to this instance of a matmul operation.
    call->setAttr(b.getStringAttr("uid"), b.getI64IntegerAttr(uid_.uid()));

    // Copy backend specific attributes.
    call->setAttr(b.getStringAttr("algorithm"), op.getAlgorithmAttr());
    call->setAttr(b.getStringAttr("alpha_imag"), op.getAlphaImagAttr());
    call->setAttr(b.getStringAttr("alpha_real"), op.getAlphaRealAttr());
    call->setAttr(b.getStringAttr("beta"), op.getBetaAttr());
    call->setAttr(b.getStringAttr("dot_dims"), op.getDotDimensionNumbers());
    call->setAttr(b.getStringAttr("epilogue"), op.getEpilogueAttr());

    // TODO(ezhulenev): Today we can't pass an array of enum attributes to the
    // custom call. Also we do not have a corresponding precision enum on the
    // SE/XLA side, so we encode it as an i32 array (tensor).
    if (auto precisions = op.getPrecisionConfig()) {
      llvm::SmallVector<int32_t> values;
      for (auto precision : *precisions) {
        auto value = precision.cast<mhlo::PrecisionAttr>().getValue();
        values.push_back(static_cast<int32_t>(value));
      }
      call->setAttr(b.getStringAttr("precision"), b.getI32TensorAttr(values));
    } else {
      call->setAttr(b.getStringAttr("precision"), b.getI32TensorAttr({0, 0}));
    }

    // Erase the original matmul operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  GemmUidGenerator& uid_;
};

// -------------------------------------------------------------------------- //

template <typename Conv>
class ConvOpLowering : public OpRewritePattern<Conv> {
 private:
  static StringRef CustomCallTarget(ConvForwardOp) {
    return "xla.gpu.conv.forward";
  }
  static StringRef CustomCallTarget(ConvForwardFusedOp) {
    return "xla.gpu.conv.forward.fused";
  }
  static StringRef CustomCallTarget(ConvForwardFusedSideInputOp) {
    return "xla.gpu.conv.forward.fused.side_input";
  }
  static StringRef CustomCallTarget(ConvBackwardFilterOp) {
    return "xla.gpu.conv.backward.filter";
  }
  static StringRef CustomCallTarget(ConvBackwardInputOp) {
    return "xla.gpu.conv.backward.input";
  }

 public:
  explicit ConvOpLowering(MLIRContext* ctx) : OpRewritePattern<Conv>(ctx) {}

  LogicalResult matchAndRewrite(Conv op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->template getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(CustomCallTarget(op)));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), CustomCallTarget(op),
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert Conv to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    // Helper functins to copy attributes from the conv op to the custom call.
    auto set_attr = [&](StringRef name, Attribute attr) {
      call->setAttr(b.getStringAttr(name), attr);
    };

    auto set_xi64 = [&](StringRef name, Optional<DenseIntElementsAttr> attr) {
      SmallVector<int64_t> values;
      if (attr.has_value())
        values = llvm::to_vector(attr->getValues<int64_t>());
      set_attr(name, b.getI64TensorAttr(values));
    };

    // Convert `BoolElementsAttr` to i64 before passing to the runtime.
    // TODO(ezhulenev): Allow passing boolean tensors to the JitRt custom calls.
    auto set_xi1 = [&](StringRef name, Optional<DenseElementsAttr> attr) {
      SmallVector<int64_t> values;
      if (attr.has_value())
        values.assign(attr->getValues<bool>().begin(),
                      attr->getValues<bool>().end());
      set_attr(name, b.getI64TensorAttr(values));
    };

    // Copy dimension number attributes.
    call->setAttr(b.getStringAttr("conv_dims"), op.getDimensionNumbers());

    // Copy convolution window attributes.
    set_xi1("window_reversal", op.getWindowReversal());
    set_xi64("window_strides", op.getWindowStrides());
    set_xi64("lhs_dilation", op.getLhsDilation());
    set_xi64("rhs_dilation", op.getRhsDilation());
    set_xi64("padding", op.getPadding());

    // Copy backend config.
    call->setAttr(b.getStringAttr("backend_config"), op.getBackendConfig());

    // Copy remaining attributes.
    set_attr("feature_group_count", op.getFeatureGroupCountAttr());
    set_attr("result_scale", op.getResultScaleAttr());

    // Copy attributes specific for fused convolutions.
    if (auto fused = dyn_cast<ConvForwardFusedOp>(op.getOperation())) {
      call->setAttr(b.getStringAttr("activation_mode"),
                    fused.getActivationModeAttr());
    }

    // Copy attributes specific for fused convolutions with side input.
    if (auto fused = dyn_cast<ConvForwardFusedSideInputOp>(op.getOperation())) {
      call->setAttr(b.getStringAttr("activation_mode"),
                    fused.getActivationModeAttr());
      set_attr("side_input_scale", fused.getSideInputScaleAttr());
    }

    // Erase the original conv operation.
    rewriter.eraseOp(op);

    return success();
  }
};

class ConvForwardOpLowering : public ConvOpLowering<ConvForwardOp> {
 public:
  using ConvOpLowering::ConvOpLowering;
};

class ConvForwardFusedOpLowering : public ConvOpLowering<ConvForwardFusedOp> {
 public:
  using ConvOpLowering::ConvOpLowering;
};

class ConvBackwardFilterOpLowering
    : public ConvOpLowering<ConvBackwardFilterOp> {
 public:
  using ConvOpLowering::ConvOpLowering;
};

class ConvBackwardInputOpLowering : public ConvOpLowering<ConvBackwardInputOp> {
 public:
  using ConvOpLowering::ConvOpLowering;
};

class ConvForwardFusedSideInputOpLowering
    : public ConvOpLowering<ConvForwardFusedSideInputOp> {
 public:
  using ConvOpLowering::ConvOpLowering;
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
    op.getCond().cloneInto(&loop.getBefore(), mapping);
    op.getBody().cloneInto(&loop.getAfter(), mapping);

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

class CaseOpLowering : public OpRewritePattern<CaseOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CaseOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Copy index buffer to the host ...
    auto index_type = op.getIndex().getType().dyn_cast<MemRefType>();
    Value index_on_host = b.create<memref::AllocaOp>(index_type);
    b.create<gpu::MemcpyOp>(TypeRange(),
                            ValueRange({index_on_host, op.getIndex()}));

    // Get the index value from the buffer.
    Value index = b.create<memref::LoadOp>(index_type.getElementType(),
                                           index_on_host, ValueRange());

    bool is_predicate = index_type.getElementType().isInteger(1);

    // For binary index (predicate) convert i1 to i32 index.
    if (is_predicate) {
      Value c0 = b.create<ConstantOp>(b.getI32IntegerAttr(0));
      Value c1 = b.create<ConstantOp>(b.getI32IntegerAttr(1));
      index = b.create<arith::SelectOp>(index, c0, c1);
    }

    // For integer index make sure that it is within range.
    if (!is_predicate) {
      unsigned n = op.getNumRegions() - 1;
      Value c0 = b.create<ConstantOp>(b.getI32IntegerAttr(0));
      Value cN = b.create<ConstantOp>(b.getI32IntegerAttr(n));

      Value too_small = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::slt, index, c0);
      Value too_large = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::sgt, index, cN);

      Value out_of_range = b.create<arith::OrIOp>(too_small, too_large);
      index = b.create<arith::SelectOp>(out_of_range, cN, index);
    }

    // Split block right at the case operation.
    Block* cont = rewriter.splitBlock(op->getBlock(), op->getIterator());
    Block* orig = cont->getPrevNode();

    // Prepare case destinations for the `scf.switch` operation.
    llvm::SmallVector<llvm::APInt> case_values;
    llvm::SmallVector<Block*> case_blocks;
    llvm::SmallVector<ValueRange> case_operands;

    // Create blocks from each of the case regions.
    for (Region& region : op->getRegions()) {
      // Move `lmhlo.case` block before the continuation.
      Block& block = region.front();
      block.moveBefore(cont);

      // Erase original `lmhlo.terminator`.
      rewriter.eraseOp(block.getTerminator());

      // Branch into the continuation block.
      b.setInsertionPointToEnd(&block);
      b.create<cf::BranchOp>(cont);

      // Add a `cf.switch` case.
      int32_t idx = case_blocks.size();
      case_values.push_back(b.getI32IntegerAttr(idx).getValue());
      case_blocks.push_back(&block);
      case_operands.push_back({});
    }

    // Replace `lmhlo.case` with a `cf.switch` operation on the host.
    b.setInsertionPointToEnd(orig);
    b.create<cf::SwitchOp>(index, cont, ValueRange(), case_values, case_blocks,
                           case_operands);

    // Erase the original case operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

class CustomCallOpLowering : public OpRewritePattern<CustomCallOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.custom_call";

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(kCustomCallTarget));

    // By default all operands passed to the custom call handler.
    llvm::SmallVector<Value> operands = op.getOperands();

    // If custom call has target arguments mapping, then we need to pass empty
    // memrefs in place of holes.
    if (op.getTargetArgMapping().has_value()) {
      auto mapping = *op.getTargetArgMapping();
      int64_t num_args = mapping.getNumArgs();
      int64_t num_results = mapping.getNumResults();

      // We represent holes as empty i8 memrefs.
      Value hole = b.create<AllocaOp>(MemRefType::get({0}, b.getI8Type()));
      operands = llvm::SmallVector<Value>(num_args + num_results, hole);

      // Update operands to mapped custom call arguments.
      auto args = mapping.getArgsToTargetArgs();
      for (const auto& indexed : llvm::enumerate(args))
        operands[indexed.value()] = op.getArgs()[indexed.index()];

      // Update operands to mapped custom call results.
      auto res = mapping.getResultsToTargetResults();
      for (const auto& indexed : llvm::enumerate(res))
        operands[num_args + indexed.value()] = op.getOutput()[indexed.index()];
    }

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, TypeRange(ValueRange(operands)), TypeRange());

    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), kCustomCallTarget,
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(op->getParentOfType<ModuleOp>());
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Call the runtime intrinsic with the original operands.
    auto call =
        rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(), operands);

    // Pass attributes to the custom call handler.
    auto set_attr = [&](StringRef name, Attribute attr) {
      call->setAttr(b.getStringAttr(name), attr);
    };

    set_attr("api_version", op.getApiVersionAttr());
    set_attr("backend_config", op.getBackendConfigAttr());
    set_attr("call_target_name", op.getCallTargetNameAttr());

    // Erase the original infeed/outfeed operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

class FftOpLowering : public OpRewritePattern<FftOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.fft";

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FftOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(kCustomCallTarget));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), kCustomCallTarget,
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert Fft to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    // Copy backend specific attributes.
    call->setAttr(b.getStringAttr("fft_length"), op.getFftLengthAttr());
    call->setAttr(b.getStringAttr("fft_type"), op.getFftTypeAttr());

    // Erase the original Fft operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

class CholeskyOpLowering : public OpRewritePattern<CholeskyOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.cholesky";

 public:
  explicit CholeskyOpLowering(MLIRContext* ctx)
      : OpRewritePattern<CholeskyOp>(ctx) {}

  LogicalResult matchAndRewrite(CholeskyOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(kCustomCallTarget));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), kCustomCallTarget,
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert Cholesky to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    const auto& dims =
        op.getInput().getType().cast<mlir::MemRefType>().getShape();
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
    call->setAttr(b.getStringAttr("is_lower"), op.getIsLowerAttr());

    // Erase the original Cholesky operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

// We assign unique id to all collective operations in the module, so that we
// can efficiently access per-op state at run time. Exception to this rule are
// asynchronous collective operations, that share the same unique id by the pair
// of corresponding `start` and `done` operations.
//
// Asynchronous collective operations pass HLO Token to represent the dependency
// between the `Start` and `Done` operations. When we lower to JitRt custom
// calls we rely on assigning each unique pair of `Start` and `Done` operations
// a unique event id, and use shared "context" owned by the GpuExecutable to
// pass Gpu events from `Start` to `Done` custom call handlers.
//
// TODO(ezhulenev): Once JitRt custom calls support returning values, we should
// explicitly return event id from the `Start` custom call, and pass it to the
// `Done` custom call. Longer term this should become an `!async.token` and rely
// on JitRt asynchonous execution.
class CollectiveUidGenerator {
 public:
  CollectiveUidGenerator() : cnt_(0) {}

  // Assings a unique event id to the pair of start and done operations.
  int32_t AssignUid(AllReduceStartOp start, AllReduceDoneOp done) {
    int32_t id = next();
    uids_[start] = id;
    uids_[done] = id;
    return id;
  }

  FailureOr<int32_t> AssignedUid(Operation* op) {
    // Async operations must be assigned uid ahead of time.
    if (isa<AllReduceStartOp, AllReduceDoneOp>(op)) {
      auto it = uids_.find(op);
      if (it == uids_.end()) return failure();
      return it->second;
    }
    // For every other operation we just assign a next id.
    return next();
  }

 private:
  int32_t next() { return cnt_++; }

  int32_t cnt_;
  llvm::DenseMap<Operation*, int32_t> uids_;
};

template <typename CollectiveOp>
class CollectiveOpLowering : public OpRewritePattern<CollectiveOp> {
 private:
  static StringRef CustomCallTarget(AllGatherOp) {
    return "xla.gpu.all_gather";
  }
  static StringRef CustomCallTarget(AllReduceOp) {
    return "xla.gpu.all_reduce";
  }
  static StringRef CustomCallTarget(AllReduceStartOp) {
    return "xla.gpu.all_reduce_start";
  }
  static StringRef CustomCallTarget(ReduceScatterOp) {
    return "xla.gpu.reduce_scatter";
  }
  static StringRef CustomCallTarget(AllToAllOp) { return "xla.gpu.all_to_all"; }
  static StringRef CustomCallTarget(CollectivePermuteOp) {
    return "xla.gpu.collective_permute";
  }

  template <typename ReduceOrGatherOp>
  static xla::gpu::NcclCollectiveConfig GetNcclCollectiveConfig(
      ReduceOrGatherOp op, int /*replica_count*/, int /*num_partitions*/) {
    return xla::gpu::GetNcclCollectiveConfigForMlir(op,
                                                    op.getUseGlobalDeviceIds());
  }
  static xla::gpu::NcclCollectiveConfig GetNcclCollectiveConfig(
      AllToAllOp op, int /*replica_count*/, int /*num_partitions*/) {
    // TODO(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
    // attribute and it should be removed.
    return xla::gpu::GetNcclCollectiveConfigForMlir(op, std::nullopt);
  }
  static xla::gpu::NcclCollectiveConfig GetNcclCollectiveConfig(
      CollectivePermuteOp op, int replica_count, int num_partitions) {
    return xla::gpu::NcclCollectivePermuteThunk::GetNcclCollectivePermuteConfig(
               op, replica_count, num_partitions)
        .config;
  }

  template <typename NonCollectivePermuteOp>
  static LogicalResult TryDegenerateToMemCopy(
      NonCollectivePermuteOp op, const xla::gpu::NcclCollectiveConfig& config,
      int replica_count, int num_partitions, PatternRewriter& rewriter) {
    if (!config.IsDegenerate(replica_count, num_partitions)) {
      return failure();
    }

    for (int64_t i = 0; i < op.getInputs().size(); i++) {
      rewriter.create<gpu::MemcpyOp>(
          op.getLoc(), TypeRange(),
          ValueRange({op.getOutputs()[i], op.getOperands()[i]}));
    }
    return success();
  }
  static LogicalResult TryDegenerateToMemCopy(
      CollectivePermuteOp op, const xla::gpu::NcclCollectiveConfig& config,
      int replica_count, int num_partitions, PatternRewriter& rewriter) {
    if (!xla::gpu::NcclCollectivePermuteThunk::IsDegenerate(op, replica_count,
                                                            num_partitions)) {
      return failure();
    }

    rewriter.create<gpu::MemcpyOp>(
        op.getLoc(), TypeRange(),
        ValueRange({op.getOutput(), op.getOperand()}));
    return success();
  }

  static bool CanImplement(AllGatherOp op) {
    return xla::gpu::NcclAllGatherThunk::CanImplement(op);
  }
  static bool CanImplement(AllReduceOp op) {
    return xla::gpu::NcclAllReduceThunk::CanImplement(op);
  }
  static bool CanImplement(AllReduceStartOp op) {
    return xla::gpu::NcclAllReduceStartThunk::CanImplement(op);
  }
  static bool CanImplement(ReduceScatterOp op) {
    return xla::gpu::NcclReduceScatterThunk::CanImplement(op);
  }
  static bool CanImplement(AllToAllOp op) {
    return xla::gpu::NcclAllToAllThunk::CanImplement(op);
  }
  static bool CanImplement(CollectivePermuteOp op) {
    return xla::gpu::NcclCollectivePermuteThunk::CanImplement(op);
  }

  template <typename ReduceOp>
  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b, ReduceOp op,
                                        CallOp call) {
    std::optional<xla::ReductionKind> reduction_kind =
        xla::gpu::NcclAllReduceThunkBase::MatchAllReduceComputation(
            op.getComputation());
    if (!reduction_kind.has_value())
      return op.emitOpError()
             << "Failed to determine reduction computation for AllReduce";

    call->setAttr(
        b.getStringAttr("reduction_kind"),
        b.getI64IntegerAttr(static_cast<int64_t>(reduction_kind.value())));
    return success();
  }
  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b, AllGatherOp op,
                                        CallOp call) {
    return success();
  }
  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b, AllToAllOp op,
                                        CallOp call) {
    call->setAttr(b.getStringAttr("has_split_dimension"),
                  b.getBoolAttr(op.getSplitDimension().has_value()));
    return success();
  }
  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b,
                                        CollectivePermuteOp op, CallOp call) {
    auto source_target_pairs_or =
        xla::ConvertNx2Attribute(op.getSourceTargetPairs());
    if (!source_target_pairs_or.ok()) {
      return op.emitOpError()
             << source_target_pairs_or.status().error_message();
    }

    // Pass an array of pairs as two vectors.
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
        std::move(source_target_pairs_or.value());
    std::vector<int64_t> source_peers, target_peers;
    source_peers.reserve(source_target_pairs.size());
    target_peers.reserve(source_target_pairs.size());
    for (const auto& source_target_pair : source_target_pairs) {
      source_peers.push_back(source_target_pair.first);
      target_peers.push_back(source_target_pair.second);
    }

    auto source_peers_attr = b.getI64TensorAttr(source_peers);
    auto target_peers_attr = b.getI64TensorAttr(target_peers);
    call->setAttr(b.getStringAttr("source_peers"), source_peers_attr);
    call->setAttr(b.getStringAttr("target_peers"), target_peers_attr);
    return success();
  }

 public:
  explicit CollectiveOpLowering(MLIRContext* ctx, CollectiveUidGenerator& uid)
      : OpRewritePattern<CollectiveOp>(ctx), uid_(uid) {}

  LogicalResult matchAndRewrite(CollectiveOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->template getParentOfType<ModuleOp>();

    // Construct an NCCL collective config from the parent func attributes.
    FuncOp fn = op->template getParentOfType<FuncOp>();
    auto replica_count_attr = fn->getAttrOfType<IntegerAttr>("replica_count");
    auto num_partitions_attr = fn->getAttrOfType<IntegerAttr>("num_partitions");
    const int64_t replica_count = replica_count_attr.getInt();
    const int64_t num_partitions = num_partitions_attr.getInt();

    xla::gpu::NcclCollectiveConfig config =
        GetNcclCollectiveConfig(op, replica_count, num_partitions);

    // A given collective op can be degenerate if across all groups formed
    // by it are singleton. In such a case, we don't need to do any
    // communication and we can just copy the input to the output.
    if (succeeded(TryDegenerateToMemCopy(op, config, replica_count,
                                         num_partitions, rewriter))) {
      // For async collective erase all corresponding done operations.
      if (auto start = dyn_cast<AllReduceStartOp>(op.getOperation())) {
        auto users = llvm::to_vector(start.getToken().getUsers());
        llvm::for_each(users, [&](Operation* user) {
          if (isa<AllReduceDoneOp>(user)) rewriter.eraseOp(user);
        });
      }

      // Erase the original collective operation.
      rewriter.eraseOp(op);

      return success();
    }

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(CustomCallTarget(op)));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op.getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), CustomCallTarget(op),
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert collective op to a function call.
    auto call = rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                                        op.getOperands());

    if (!CanImplement(op)) {
      return op.emitOpError()
             << "Requested " << CustomCallTarget(op)
             << " not implemented on GPU; replica_count: " << replica_count
             << ", num_partitions: " << num_partitions << ", group_mode: "
             << CollectiveOpGroupModeToString(config.group_mode)
             << ", NCCL support: "
             << xla::gpu::NcclCollectiveThunk::NcclIsEnabled();
    }

    // Copy backend specific attributes.
    call->setAttr(b.getStringAttr("group_mode"),
                  b.getI64IntegerAttr(static_cast<int64_t>(config.group_mode)));
    call->setAttr(b.getStringAttr("op_id"), b.getI64IntegerAttr(config.op_id));
    // TODO(b/233930690): Pass the attribute below as a nested array.
    // Pass an array of arrays using two vectors; one specifying all the values
    // and another specifying the (ending) offsets of each array in the other
    // vector. Example: [ [10, 20, 30, 40], [50, 60], [70, 80, 90] ] turns into
    // offsets=[4, 6, 9] values=[10, 20, 30, 40, 50, 60, 70, 80, 90].
    std::vector<int64_t> replica_group_offsets;
    std::vector<int64_t> replica_group_values;
    replica_group_offsets.reserve(config.replica_groups.size());
    int replica_group_offset = 0;
    for (const auto& replica_group : config.replica_groups) {
      replica_group_offset += replica_group.replica_ids_size();
      replica_group_offsets.push_back(replica_group_offset);
      replica_group_values.reserve(replica_group_offset);
      for (auto replica_id : replica_group.replica_ids()) {
        replica_group_values.push_back(replica_id);
      }
    }
    call->setAttr(b.getStringAttr("replica_group_offsets"),
                  b.getI64TensorAttr(replica_group_offsets));
    call->setAttr(b.getStringAttr("replica_group_values"),
                  b.getI64TensorAttr(replica_group_values));

    // Assign a unique collective operation id.
    auto uid = uid_.AssignedUid(op);
    if (succeeded(uid)) {
      call->setAttr(b.getStringAttr("uid"), b.getI32IntegerAttr(*uid));
    } else {
      return op.emitOpError("failed to get a unique collective operation id");
    }

    // Set attributes specific to the type of collective operation.
    auto result = SetSpecificAttrs(b, op, call);
    if (failed(result)) return result;

    // For asynchonous start operation we need to produce a fake token, that
    // will be later removed, because corresponding `done` operation doesn't
    // have the token argument. We rely on the `unrealized_conversion_cast`
    // operation to create a fake token from the `i8` constant.
    if (auto start = dyn_cast<AllReduceStartOp>(op.getOperation())) {
      Value token = start.getToken();
      Value c0 = b.create<ConstantOp>(b.getI8IntegerAttr(0));
      auto fake = b.create<UnrealizedConversionCastOp>(token.getType(), c0);
      token.replaceAllUsesWith(fake.getResult(0));
    }

    // Erase the original collective operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  CollectiveUidGenerator& uid_;
};

class AllGatherOpLowering : public CollectiveOpLowering<AllGatherOp> {
 public:
  using CollectiveOpLowering::CollectiveOpLowering;
};

class AllReduceOpLowering : public CollectiveOpLowering<AllReduceOp> {
 public:
  using CollectiveOpLowering::CollectiveOpLowering;
};

class AllReduceStartOpLowering : public CollectiveOpLowering<AllReduceStartOp> {
 public:
  using CollectiveOpLowering::CollectiveOpLowering;
};

class ReduceScatterOpLowering : public CollectiveOpLowering<ReduceScatterOp> {
 public:
  using CollectiveOpLowering::CollectiveOpLowering;
};

class AllToAllOpLowering : public CollectiveOpLowering<AllToAllOp> {
 public:
  using CollectiveOpLowering::CollectiveOpLowering;
};

class CollectivePermuteOpLowering
    : public CollectiveOpLowering<CollectivePermuteOp> {
 public:
  using CollectiveOpLowering::CollectiveOpLowering;
};

class AllReduceDoneOpLowering : public OpRewritePattern<AllReduceDoneOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.all_reduce_done";

 public:
  explicit AllReduceDoneOpLowering(MLIRContext* ctx,
                                   CollectiveUidGenerator& uid)
      : OpRewritePattern<AllReduceDoneOp>(ctx), uid_(uid) {}

  LogicalResult matchAndRewrite(AllReduceDoneOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(kCustomCallTarget));

    // For done operation we drop the token argument and communicate async event
    // dependency through the `uid` attribute.
    llvm::SmallVector<Value> operands = op.getOperands().drop_front();

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, TypeRange(ValueRange(operands)), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), kCustomCallTarget,
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert AllReduceDone to a function call.
    auto call =
        rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(), operands);

    // Assign a unique collective operation id.
    auto uid = uid_.AssignedUid(op);
    if (succeeded(uid)) {
      call->setAttr(b.getStringAttr("uid"), b.getI32IntegerAttr(*uid));
    } else {
      return op.emitOpError("failed to get a unique collective operation id");
    }

    // Erase the original AllReduceDone operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  CollectiveUidGenerator& uid_;
};

// -------------------------------------------------------------------------- //

template <typename IdOp>
class IdOpLowering : public OpRewritePattern<IdOp> {
 private:
  static StringRef CustomCallTarget(ReplicaIdOp) {
    return "xla.gpu.replica_id";
  }
  static StringRef CustomCallTarget(PartitionIdOp) {
    return "xla.gpu.partition_id";
  }

 public:
  explicit IdOpLowering(MLIRContext* ctx) : OpRewritePattern<IdOp>(ctx) {}

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = this->getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModuleOp module = op->template getParentOfType<ModuleOp>();

    // Custom call target.
    NamedAttribute target(b.getStringAttr(kDirectCustomCall),
                          b.getStringAttr(CustomCallTarget(op)));

    // Create a custom call function declaration.
    auto custom_call_type =
        FunctionType::get(ctx, op->getOperandTypes(), TypeRange());
    auto custom_call_attrs = ArrayRef<NamedAttribute>(target);
    auto custom_call = FuncOp::create(op.getLoc(), CustomCallTarget(op),
                                      custom_call_type, custom_call_attrs);
    custom_call.setPrivate();

    SymbolTable sym_table(module);
    auto inserted = sym_table.insert(custom_call);
    rewriter.notifyOperationInserted(custom_call);

    // Convert ReplicaId to a function call.
    rewriter.create<CallOp>(op.getLoc(), inserted, TypeRange(),
                            op->getOperands());

    // Erase the original ReplicaId operation.
    rewriter.eraseOp(op);

    return success();
  }
};

class ReplicaIdOpLowering : public IdOpLowering<ReplicaIdOp> {
 public:
  using IdOpLowering::IdOpLowering;
};

class PartitionIdOpLowering : public IdOpLowering<PartitionIdOp> {
 public:
  using IdOpLowering::IdOpLowering;
};

// -------------------------------------------------------------------------- //

void ConvertLmhloGpuToJitRtPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Convert lmhlo_gpu operations to JitRt gpu runtime custom calls.
  RewritePatternSet patterns(ctx);

  // Each unique Gemm/Matmul operation in the module will get assigned a uid.
  GemmUidGenerator gemm_uid;
  patterns.insert<GemmOpLowering, CublasLtMatmulOpLowering>(ctx, gemm_uid);

  // Assign shared unique id to each unique pair of async start-done operations,
  // all other collective operations will get assigned uid.
  CollectiveUidGenerator collective_uid;
  auto walked = module.walk([&](AllReduceStartOp start) -> WalkResult {
    Value token = start.getToken();

    // We expect the token to be consumed just once.
    if (!token.hasOneUse()) return start.emitOpError("token has multiple uses");

    // Token must be consumed by the corresponding done operation.
    auto done = dyn_cast<AllReduceDoneOp>(*token.getUsers().begin());
    if (!done) return start.emitOpError("illegal token user");

    collective_uid.AssignUid(start, done);
    return WalkResult::advance();
  });
  if (walked.wasInterrupted()) return signalPassFailure();

  // Patterns for collective operations.
  patterns.insert<AllGatherOpLowering, AllReduceOpLowering,
                  AllReduceStartOpLowering, AllToAllOpLowering,
                  CollectivePermuteOpLowering, ReduceScatterOpLowering>(
      ctx, collective_uid);

  // Patterns for every other Gpu operation.
  patterns
      .insert<FftOpLowering, CholeskyOpLowering, PartitionIdOpLowering,
              ReplicaIdOpLowering, WhileOpLowering, CaseOpLowering,
              CustomCallOpLowering, TerminatorOpLowering, ConvForwardOpLowering,
              ConvForwardFusedOpLowering, ConvForwardFusedSideInputOpLowering,
              ConvBackwardFilterOpLowering, ConvBackwardInputOpLowering>(ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();

  // TODO(ezhulenev): We must run `done` op lowering after the `start` op
  // lowering to ensure that all redundant collective operations will be
  // safely replaced by a `memcpy` operations. We should find a better way to
  // achieve this goal.
  {
    RewritePatternSet patterns(ctx);
    patterns.insert<AllReduceDoneOpLowering>(ctx, collective_uid);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertLmhloGpuToJitRtPass() {
  return std::make_unique<ConvertLmhloGpuToJitRtPass>();
}

void populateLmhloToJitRtPasses(mlir::OpPassManager& pm,
                                xla::gpu::ThunkSequence* thunk_sequence) {
  // Small global constants will be embedded into the device modules.
  pm.addPass(createConvertLmhloToGpuBinaryPass(thunk_sequence));

  // Convert global memrefs corresponding to constant arguments.
  pm.addPass(xla::gpu::createConvertMemrefGetGlobalToArgPass());
  pm.addPass(createSymbolDCEPass());  // Clean up unused global constants.

  // Lower all Gpu operations to the JitRt Gpu runtime intrinsics.
  pm.addPass(createConvertLmhloGpuToJitRtPass());
  pm.addPass(xla::gpu::createConvertGpuToGpuRuntimePass());
  pm.addPass(xla::gpu::createConvertLmhloToGpuRuntimePass());
}

void registerLmhloToJitRtPasses() {
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
