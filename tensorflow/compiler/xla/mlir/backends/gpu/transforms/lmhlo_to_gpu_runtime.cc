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

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/uid_generator.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTLMHLOTOGPURUNTIMEPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::MemcpyOp;

using mlir::lmhlo::CaseOp;
using mlir::lmhlo::CustomCallOp;
using mlir::lmhlo::FftOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;
using mlir::lmhlo::TerminatorOp;
using mlir::lmhlo::WhileOp;

using xla::runtime::AppendCustomCallAttrs;
using xla::runtime::CustomCallDeclarations;

// helper template to check T is any of the types listed in Ts.
template <typename T, typename... Ts>
inline constexpr bool is_any = std::disjunction_v<std::is_same<T, Ts>...>;

class ConvertLmhloToGpuRuntimePass
    : public impl::ConvertLmhloToGpuRuntimePassBase<
          ConvertLmhloToGpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }
};

//===----------------------------------------------------------------------===//

class TerminatorOpLowering : public OpRewritePattern<TerminatorOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TerminatorOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//

template <typename IoFeedOp>
class IoFeedOpLowering : public OpRewritePattern<IoFeedOp> {
  static StringRef Target(InfeedOp) { return "xla.gpu.infeed"; }
  static StringRef Target(OutfeedOp) { return "xla.gpu.outfeed"; }

 public:
  IoFeedOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern<IoFeedOp>(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(IoFeedOp op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, Target(op), op);

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("config"), op.getConfigAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, custom_call_attrs);

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

class InfeedOpLowering : public IoFeedOpLowering<InfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

class OutfeedOpLowering : public IoFeedOpLowering<OutfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

//===----------------------------------------------------------------------===//

class CustomCallOpLowering : public OpRewritePattern<CustomCallOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.custom_call";

 public:
  CustomCallOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  // Rewrite custom call with `API_VERSION_TYPED_FFI` version into XLA runtime
  // custom calls bypassing custom call adaptor.
  LogicalResult rewriteTypedCustomCall(CustomCallOp op,
                                       PatternRewriter& rewriter) const {
    // TODO(ezhulenev): Support target arg mapping, or explain why we do not
    // need them for typed custom calls.
    if (op.getTargetArgMapping())
      return op.emitOpError(
          "API_VERSION_TYPED_FFI custom calls do not "
          "support target arg mapping");

    // Create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee =
        custom_calls_.GetOrCreate(b, op.getCallTargetName(), op);
    callee->setAttr("rt.dynamic", UnitAttr::get(b.getContext()));

    // Forward backend config to the custom call implementation.
    auto dict = op.getBackendConfig()
                    ? op.getBackendConfig()->cast<mlir::DictionaryAttr>()
                    : nullptr;
    llvm::SmallVector<NamedAttribute> backend_config(dict.begin(), dict.end());

    // Call the custom call function forwarding user-defined attributes.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, backend_config);

    return success();
  }

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    // Typed custom calls lowered directly to XLA runtime custom calls.
    if (op.getApiVersion() == mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI)
      return rewriteTypedCustomCall(op, rewriter);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // By default all operands passed to the custom call handler.
    llvm::SmallVector<Value> operands = op.getOperands();

    // If custom call has target arguments mapping, then we need to pass `i64`
    // scalars in place of holes to detect them in custom call handler.
    //
    // TODO(ezhulenev): We need an `xla` dialect to model Xla framework
    // semantics including holes for custom call. As a work around we pass `i64`
    // values because xla custom call do not support scalar arguments, and we
    // can disambiguate holes from buffers.
    if (op.getTargetArgMapping().has_value()) {
      auto mapping = *op.getTargetArgMapping();
      int64_t num_args = mapping.getNumArgs();
      int64_t num_results = mapping.getNumResults();

      // We represent holes as an arbitrary `i64` constant.
      Value hole = b.create<arith::ConstantOp>(b.getI64IntegerAttr(-1));
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
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCustomCallTarget, TypeRange(ValueRange(operands)), TypeRange());

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("api_version"), op.getApiVersionAttr()},
        {b.getStringAttr("backend_config"), op.getBackendConfigAttr()},
        {b.getStringAttr("call_target_name"), op.getCallTargetNameAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), operands);
    AppendCustomCallAttrs(call, custom_call_attrs);

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class FftOpLowering : public OpRewritePattern<FftOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.fft";

 public:
  FftOpLowering(MLIRContext* ctx, UidGenerator& uid,
                CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), uid_(uid), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(FftOp op,
                                PatternRewriter& rewriter) const override {
    // Create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, kCustomCallTarget, op);

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("fft_length"), op.getFftLengthAttr()},
        {b.getStringAttr("fft_type"), op.getFftTypeAttr()},
        {b.getStringAttr("uid"), b.getI64IntegerAttr(uid_.uid())}};

    // Convert Fft to a function call.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, custom_call_attrs);
    return success();
  }

 private:
  UidGenerator& uid_;
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class CaseOpLowering : public OpRewritePattern<CaseOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CaseOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Copy index buffer to the host ...
    auto index_type = op.getIndex().getType().dyn_cast<MemRefType>();

    // Always create an `alloca` in the parent function entry block.
    // See: https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
    Value index_on_host = [&]() -> Value {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&op->getParentOfType<func::FuncOp>().front());
      return b.create<memref::AllocaOp>(index_type);
    }();

    b.create<MemcpyOp>(TypeRange(), ValueRange({index_on_host, op.getIndex()}));

    // Get the index value from the buffer.
    Value index = b.create<memref::LoadOp>(index_type.getElementType(),
                                           index_on_host, ValueRange());

    bool is_predicate = index_type.getElementType().isInteger(1);

    // For binary index (predicate) convert i1 to i32 index.
    if (is_predicate) {
      Value c0 = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
      Value c1 = b.create<arith::ConstantOp>(b.getI32IntegerAttr(1));
      index = b.create<arith::SelectOp>(index, c0, c1);
    }

    // For integer index make sure that it is within range.
    if (!is_predicate) {
      unsigned n = op.getNumRegions() - 1;
      Value c0 = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
      Value cN = b.create<arith::ConstantOp>(b.getI32IntegerAttr(n));

      Value too_small = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::slt, index, c0);
      Value too_large = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::sgt, index, cN);

      Value out_of_range = b.create<arith::OrIOp>(too_small, too_large);
      index = b.create<arith::SelectOp>(out_of_range, cN, index);
    }

    // Wrap the CFG constructed from the `lmhlo.case` operation in an
    // `scf.execute_region` operation, so that we do not introduce the CFG
    // into regions that expect a single block (e.g. inside the loop body).
    auto execute = b.create<scf::ExecuteRegionOp>(TypeRange());

    // Add an entry block to the execute region operation.
    Block& entry = execute.getRegion().emplaceBlock();

    // Create a block with `scf.yield` terminator.
    Block& yield = execute.getRegion().emplaceBlock();
    b.setInsertionPointToStart(&yield);
    b.create<scf::YieldOp>();

    // Prepare case destinations for the `scf.switch` operation.
    llvm::SmallVector<llvm::APInt> case_values;
    llvm::SmallVector<Block*> case_blocks;
    llvm::SmallVector<ValueRange> case_operands;

    // Create blocks from each of the case regions.
    for (Region& region : op->getRegions()) {
      // Move `lmhlo.case` block into the execute region.
      Block& block = region.front();
      block.moveBefore(&yield);

      // Erase original `lmhlo.terminator`.
      rewriter.eraseOp(block.getTerminator());

      // Branch into the yield block.
      b.setInsertionPointToEnd(&block);
      b.create<cf::BranchOp>(&yield);

      // Add a `cf.switch` case.
      int32_t idx = case_blocks.size();
      case_values.push_back(b.getI32IntegerAttr(idx).getValue());
      case_blocks.push_back(&block);
      case_operands.push_back({});
    }

    // Create a `cf.switch` operation in the execute region entry block.
    b.setInsertionPointToEnd(&entry);
    b.create<cf::SwitchOp>(index, &yield, ValueRange(), case_values,
                           case_blocks, case_operands);

    // Erase the original case operation.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//

class WhileOpLowering : public OpRewritePattern<WhileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  // Rewrite while loop with known trip count to `scf.for` operation.
  LogicalResult rewriteForLoop(WhileOp op, PatternRewriter& rewriter) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value lb = b.create<arith::ConstantIndexOp>(0);
    Value ub = b.create<arith::ConstantIndexOp>(*op.getTripCount());
    Value c1 = b.create<arith::ConstantIndexOp>(1);

    // Create an `scf.for` loop in place of `lmhlo.while` loop.
    auto loop = b.create<scf::ForOp>(lb, ub, c1, ValueRange());

    // Move body region into the new loop operation.
    IRMapping mapping;
    rewriter.eraseOp(op.getBody().front().getTerminator());
    rewriter.inlineBlockBefore(&op.getBody().front(),
                               loop.getLoopBody().front().getTerminator());

    // Erase the original while loop.
    rewriter.eraseOp(op);

    return success();
  }

  // Rewrite while loop with unknown trip count to `scf.while` operation.
  LogicalResult rewriteWhileLoop(WhileOp op, PatternRewriter& rewriter) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Create an `scf.while` loop in place of `lmhlo.while` loop.
    auto loop = b.create<scf::WhileOp>(TypeRange(), ValueRange());

    // Predicate buffer placed on the device.
    Value pred = op.getOperand(0);

    // Inline condition and body regions into the new loop operation.
    IRMapping mapping;
    rewriter.inlineRegionBefore(op.getCond(), loop.getBefore(),
                                loop.getBefore().begin());
    rewriter.inlineRegionBefore(op.getBody(), loop.getAfter(),
                                loop.getAfter().begin());

    {  // Replace loop condition terminator.
      auto* terminator = loop.getBefore().back().getTerminator();
      b.setInsertionPointAfter(terminator);

      auto i1 = b.getI1Type();

      // Always create an `alloca` in the parent function entry block.
      // See: https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
      Value pred_on_host = [&]() -> Value {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(
            &op->getParentOfType<func::FuncOp>().front());
        return b.create<memref::AllocaOp>(MemRefType::get({}, i1));
      }();

      // Copy predicate buffer to the host ...
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

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    assert(op.getNumOperands() == 1 && "expected single lmhlo.while operand");
    return op.getTripCount().has_value() ? rewriteForLoop(op, rewriter)
                                         : rewriteWhileLoop(op, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Collective operations lowerings.
//===----------------------------------------------------------------------===//

using mlir::lmhlo::AllGatherOp;
using mlir::lmhlo::AllReduceOp;
using mlir::lmhlo::AllToAllOp;
using mlir::lmhlo::CollectivePermuteOp;
using mlir::lmhlo::PartitionIdOp;
using mlir::lmhlo::ReduceScatterOp;
using mlir::lmhlo::ReplicaIdOp;
using mlir::lmhlo_gpu::AllGatherDoneOp;
using mlir::lmhlo_gpu::AllGatherStartOp;
using mlir::lmhlo_gpu::AllReduceDoneOp;
using mlir::lmhlo_gpu::AllReduceStartOp;
using mlir::lmhlo_gpu::AllToAllDoneOp;
using mlir::lmhlo_gpu::AllToAllStartOp;
using mlir::lmhlo_gpu::CollectivePermuteDoneOp;
using mlir::lmhlo_gpu::CollectivePermuteStartOp;
using mlir::lmhlo_gpu::ReduceScatterDoneOp;
using mlir::lmhlo_gpu::ReduceScatterStartOp;

// We assign unique id to all collective operations in the module, so that we
// can efficiently access per-op state at run time. Exception to this rule are
// asynchronous collective operations, that share the same unique id by the pair
// of corresponding `start` and `done` operations.
//
// Asynchronous collective operations pass HLO Token to represent the dependency
// between the `Start` and `Done` operations. When we lower to XLA runtime
// custom calls we rely on assigning each unique pair of `Start` and `Done`
// operations a unique event id, and use shared "context" owned by the
// GpuExecutable to pass Gpu events from `Start` to `Done` custom call handlers.
//
// TODO(ezhulenev): Once XLA runtime custom calls support returning values, we
// should explicitly return event id from the `Start` custom call, and pass it
// to the `Done` custom call. Longer term this should become an `!async.token`
// and rely on XLA runtime asynchonous execution.
class CollectiveUidGenerator {
 public:
  CollectiveUidGenerator() : cnt_(0) {}

  // Assings a unique event id to the pair of start and done operations.
  int32_t AssignUid(Operation* start, Operation* done) {
    int32_t id = next();
    uids_[start] = id;
    uids_[done] = id;
    return id;
  }

  FailureOr<int32_t> AssignedUid(Operation* op) {
    // Async operations must be assigned uid ahead of time.
    if (isa<AllGatherStartOp, AllGatherDoneOp, AllReduceStartOp,
            AllReduceDoneOp, AllToAllStartOp, AllToAllDoneOp,
            CollectivePermuteStartOp, CollectivePermuteDoneOp,
            ReduceScatterStartOp, ReduceScatterDoneOp>(op)) {
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
  static StringRef Target(AllGatherOp) { return "xla.gpu.all_gather"; }
  static StringRef Target(AllGatherStartOp) { return "xla.gpu.all_gather"; }

  static StringRef Target(AllReduceOp) { return "xla.gpu.all_reduce"; }
  static StringRef Target(AllReduceStartOp) { return "xla.gpu.all_reduce"; }

  static StringRef Target(AllToAllOp) { return "xla.gpu.all_to_all"; }
  static StringRef Target(AllToAllStartOp) { return "xla.gpu.all_to_all"; }

  static StringRef Target(ReduceScatterOp) { return "xla.gpu.reduce_scatter"; }
  static StringRef Target(ReduceScatterStartOp) {
    return "xla.gpu.reduce_scatter";
  }

  static StringRef Target(CollectivePermuteOp) {
    return "xla.gpu.collective_permute";
  }
  static StringRef Target(CollectivePermuteStartOp) {
    return "xla.gpu.collective_permute";
  }

  template <typename ReduceOrGatherOp>
  static NcclCollectiveConfig GetNcclCollectiveConfig(ReduceOrGatherOp op,
                                                      int /*replica_count*/,
                                                      int /*num_partitions*/) {
    return GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  }

  static NcclCollectiveConfig GetNcclCollectiveConfig(AllToAllOp op,
                                                      int /*replica_count*/,
                                                      int /*num_partitions*/) {
    // TODO(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
    // attribute and it should be removed.
    return GetNcclCollectiveConfigForMlir(op, std::nullopt);
  }

  static NcclCollectiveConfig GetNcclCollectiveConfig(CollectivePermuteOp op,
                                                      int replica_count,
                                                      int num_partitions) {
    return NcclCollectivePermuteThunk::GetNcclCollectivePermuteConfig(
               op, replica_count, num_partitions)
        .config;
  }

  static NcclCollectiveConfig GetNcclCollectiveConfig(
      CollectivePermuteStartOp op, int replica_count, int num_partitions) {
    return NcclCollectivePermuteStartThunk::GetNcclCollectivePermuteConfig(
               op, replica_count, num_partitions)
        .config;
  }

  template <typename NonCollectivePermuteOp>
  static LogicalResult TryDegenerateToMemCopy(
      NonCollectivePermuteOp op, const NcclCollectiveConfig& config,
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

  template <typename ThunkT, typename OpT>
  static LogicalResult TryDegenerateCollectivePermuteToMemCopy(
      OpT op, const NcclCollectiveConfig& config, int replica_count,
      int num_partitions, PatternRewriter& rewriter) {
    if (!ThunkT::IsDegenerate(op, replica_count, num_partitions)) {
      return failure();
    }

    rewriter.create<gpu::MemcpyOp>(
        op.getLoc(), TypeRange(),
        ValueRange({op.getOutput(), op.getOperand()}));

    return success();
  }

  static LogicalResult TryDegenerateToMemCopy(
      CollectivePermuteOp op, const NcclCollectiveConfig& config,
      int replica_count, int num_partitions, PatternRewriter& rewriter) {
    return TryDegenerateCollectivePermuteToMemCopy<NcclCollectivePermuteThunk>(
        op, config, replica_count, num_partitions, rewriter);
  }

  static LogicalResult TryDegenerateToMemCopy(
      CollectivePermuteStartOp op, const NcclCollectiveConfig& config,
      int replica_count, int num_partitions, PatternRewriter& rewriter) {
    return TryDegenerateCollectivePermuteToMemCopy<
        NcclCollectivePermuteStartThunk>(op, config, replica_count,
                                         num_partitions, rewriter);
  }

  static Status CheckImplementable(AllGatherOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclAllGatherThunk::CheckImplementable(op, replica_count,
                                                  num_partitions);
  }

  static Status CheckImplementable(AllGatherStartOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclAllGatherStartThunk::CheckImplementable(op, replica_count,
                                                       num_partitions);
  }

  static Status CheckImplementable(AllReduceOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclAllReduceThunk::CheckImplementable(op, replica_count,
                                                  num_partitions);
  }

  static Status CheckImplementable(AllReduceStartOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclAllReduceStartThunk::CheckImplementable(op, replica_count,
                                                       num_partitions);
  }

  static Status CheckImplementable(ReduceScatterOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclReduceScatterThunk::CheckImplementable(op, replica_count,
                                                      num_partitions);
  }

  static Status CheckImplementable(AllToAllOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclAllToAllThunk::CheckImplementable(op, replica_count,
                                                 num_partitions);
  }

  static Status CheckImplementable(AllToAllStartOp op, int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclAllToAllStartThunk::CheckImplementable(op, replica_count,
                                                      num_partitions);
  }

  static Status CheckImplementable(CollectivePermuteOp op,
                                   int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclCollectivePermuteThunk::CheckImplementable(op, replica_count,
                                                          num_partitions);
  }

  static Status CheckImplementable(CollectivePermuteStartOp op,
                                   int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclCollectivePermuteStartThunk::CheckImplementable(
        op, replica_count, num_partitions);
  }

  static Status CheckImplementable(ReduceScatterStartOp op,
                                   int64_t replica_count,
                                   int64_t num_partitions) {
    return NcclReduceScatterStartThunk::CheckImplementable(op, replica_count,
                                                           num_partitions);
  }

  template <typename OpT>
  static
      typename std::enable_if_t<is_any<OpT, AllReduceOp, AllReduceStartOp,
                                       ReduceScatterOp, ReduceScatterStartOp>,
                                LogicalResult>
      SetSpecificAttrs(ImplicitLocOpBuilder& b, OpT op, func::CallOp call) {
    std::optional<xla::ReductionKind> reduction_kind =
        NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
            op.getComputation());
    if (!reduction_kind.has_value())
      return op.emitOpError()
             << "Failed to determine reduction computation for AllReduce";

    call->setAttr(
        b.getStringAttr("reduction_kind"),
        b.getI64IntegerAttr(static_cast<int64_t>(reduction_kind.value())));

    return success();
  }

  template <typename OpT>
  static typename std::enable_if_t<is_any<OpT, AllGatherOp, AllGatherStartOp>,
                                   LogicalResult>
  SetSpecificAttrs(ImplicitLocOpBuilder& b, OpT op, func::CallOp call) {
    return success();
  }

  template <typename OpT>
  static typename std::enable_if_t<is_any<OpT, AllToAllOp, AllToAllStartOp>,
                                   LogicalResult>
  SetSpecificAttrs(ImplicitLocOpBuilder& b, OpT op, func::CallOp call) {
    call->setAttr(b.getStringAttr("has_split_dimension"),
                  b.getBoolAttr(op.getSplitDimension().has_value()));
    return success();
  }

  template <typename OpT>
  static typename std::enable_if_t<
      is_any<OpT, CollectivePermuteOp, CollectivePermuteStartOp>, LogicalResult>
  SetSpecificAttrs(ImplicitLocOpBuilder& b, OpT op, func::CallOp call) {
    auto source_target_pairs_or =
        ConvertNx2Attribute(op.getSourceTargetPairs());
    if (!source_target_pairs_or.ok()) {
      return op.emitOpError() << source_target_pairs_or.status().message();
    }

    // Pass an array of pairs as two vectors.
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
        std::move(source_target_pairs_or.value());
    std::vector<int64_t> source_peers;
    std::vector<int64_t> target_peers;
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

  // For async collective erase all corresponding done operations.
  template <typename StartOpT, typename DoneOpT>
  void eraseDoneOp(PatternRewriter& rewriter, CollectiveOp op) const {
    if (auto start = dyn_cast<StartOpT>(op.getOperation())) {
      auto users = llvm::to_vector(start.getToken().getUsers());
      llvm::for_each(users, [&](Operation* user) {
        if (isa<DoneOpT>(user)) rewriter.eraseOp(user);
      });
    }
  }

 public:
  CollectiveOpLowering(MLIRContext* ctx, CollectiveUidGenerator& uid,
                       CustomCallDeclarations& custom_calls)
      : OpRewritePattern<CollectiveOp>(ctx),
        uid_(uid),
        custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(CollectiveOp op,
                                PatternRewriter& rewriter) const override {
    // Construct an NCCL collective config from the parent func attributes.
    func::FuncOp fn = op->template getParentOfType<func::FuncOp>();
    auto replica_count_attr = fn->getAttrOfType<IntegerAttr>("replica_count");
    auto num_partitions_attr = fn->getAttrOfType<IntegerAttr>("num_partitions");
    const int64_t replica_count = replica_count_attr.getInt();
    const int64_t num_partitions = num_partitions_attr.getInt();

    NcclCollectiveConfig config =
        GetNcclCollectiveConfig(op, replica_count, num_partitions);

    // A given collective op can be degenerate if across all groups formed
    // by it are singleton. In such a case, we don't need to do any
    // communication and we can just copy the input to the output.
    if (succeeded(TryDegenerateToMemCopy(op, config, replica_count,
                                         num_partitions, rewriter))) {
      // For async collective erase all corresponding done operations.
      eraseDoneOp<AllGatherStartOp, AllGatherDoneOp>(rewriter, op);
      eraseDoneOp<AllReduceStartOp, AllReduceDoneOp>(rewriter, op);
      eraseDoneOp<CollectivePermuteStartOp, CollectivePermuteDoneOp>(rewriter,
                                                                     op);
      eraseDoneOp<ReduceScatterStartOp, ReduceScatterDoneOp>(rewriter, op);
      eraseDoneOp<AllToAllStartOp, AllToAllDoneOp>(rewriter, op);

      // Erase the original collective operation.
      rewriter.eraseOp(op);

      return success();
    }

    Status implementable_status =
        CheckImplementable(op, replica_count, num_partitions);
    if (!implementable_status.ok()) {
      return op.emitOpError() << implementable_status.message();
    }

    // Check that we have and assigned unique collective operation id.
    auto uid = uid_.AssignedUid(op);
    if (failed(uid)) {
      return op.emitOpError("failed to get a unique collective operation id");
    }

    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // We always drop the return value from the signature, because for
    // AllReduceStart operation we pass dependency through the collective
    // operation id.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, Target(op), TypeRange(op.getOperands()), TypeRange());

    // Convert collective op to a function call.
    auto call = rewriter.create<func::CallOp>(op.getLoc(), callee.getName(),
                                              TypeRange(), op.getOperands());

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
    call->setAttr(b.getStringAttr("uid"), b.getI32IntegerAttr(*uid));

    // Set attributes specific to the type of collective operation.
    auto result = SetSpecificAttrs(b, op, call);
    if (failed(result)) return result;

    bool is_async = false;
    // For asynchonous start operation we need to produce a fake token, that
    // will be later removed, because corresponding `done` operation doesn't
    // have a token argument. We rely on the `unrealized_conversion_cast`
    // operation to create a fake token from the `i8` constant, and on the dead
    // code elimination pass that will remove unused fake tokens.
    if constexpr (is_any<CollectiveOp, AllGatherStartOp, AllReduceStartOp,
                         AllToAllStartOp, CollectivePermuteStartOp,
                         ReduceScatterStartOp>) {
      is_async = true;
      Value token = op.getToken();
      Value c0 = b.create<arith::ConstantOp>(b.getI8IntegerAttr(0));
      auto fake = b.create<UnrealizedConversionCastOp>(token.getType(), c0);
      token.replaceAllUsesWith(fake.getResult(0));
    }
    call->setAttr(b.getStringAttr("is_async"), b.getBoolAttr(is_async));

    // Erase the original collective operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  CollectiveUidGenerator& uid_;
  CustomCallDeclarations& custom_calls_;
};

#define DEFINE_COLLECTIVE_OP_LOWERING(OP)                \
  class OP##Lowering : public CollectiveOpLowering<OP> { \
   public:                                               \
    using CollectiveOpLowering::CollectiveOpLowering;    \
  }

DEFINE_COLLECTIVE_OP_LOWERING(AllGatherOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllGatherStartOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllReduceOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllReduceStartOp);
DEFINE_COLLECTIVE_OP_LOWERING(ReduceScatterOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllToAllOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllToAllStartOp);
DEFINE_COLLECTIVE_OP_LOWERING(CollectivePermuteOp);
DEFINE_COLLECTIVE_OP_LOWERING(CollectivePermuteStartOp);
DEFINE_COLLECTIVE_OP_LOWERING(ReduceScatterStartOp);

#undef DEFINE_COLLECTIVE_OP_LOWERING

template <typename OpT, typename Derived>
class AsyncDoneOpLowering : public OpRewritePattern<OpT> {
 public:
  AsyncDoneOpLowering(MLIRContext* ctx, CollectiveUidGenerator& uid,
                      CustomCallDeclarations& custom_calls)
      : OpRewritePattern<OpT>(ctx),
        custom_call_target_(Derived::kCustomCallTarget),
        uid_(uid),
        custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, custom_call_target_,
                                                    TypeRange(), TypeRange());

    // Get a unique collective operation id.
    FailureOr<int32_t> uid = uid_.AssignedUid(op);
    if (failed(uid))
      return op.emitOpError("failed to get a unique collective operation id");

    llvm::SmallVector<NamedAttribute> custom_call_attributes = {
        {b.getStringAttr("uid"), b.getI32IntegerAttr(*uid)}};

    // Convert AllReduceDone to a function call.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(),
                                                          TypeRange());
    AppendCustomCallAttrs(call, custom_call_attributes);

    return success();
  }

 private:
  const char* custom_call_target_;
  CollectiveUidGenerator& uid_;
  CustomCallDeclarations& custom_calls_;
};

#define DEFINE_COLLECTIVE_DONE_OP_LOWERING(OP, custom_call)            \
  struct OP##Lowering : public AsyncDoneOpLowering<OP, OP##Lowering> { \
    static constexpr const char kCustomCallTarget[] = custom_call;     \
    using AsyncDoneOpLowering::AsyncDoneOpLowering;                    \
  }

DEFINE_COLLECTIVE_DONE_OP_LOWERING(AllGatherDoneOp, "xla.gpu.all_gather_done");
DEFINE_COLLECTIVE_DONE_OP_LOWERING(AllReduceDoneOp, "xla.gpu.all_reduce_done");
DEFINE_COLLECTIVE_DONE_OP_LOWERING(AllToAllDoneOp, "xla.gpu.all_to_all_done");
DEFINE_COLLECTIVE_DONE_OP_LOWERING(CollectivePermuteDoneOp,
                                   "xla.gpu.collective_permute_done");
DEFINE_COLLECTIVE_DONE_OP_LOWERING(ReduceScatterDoneOp,
                                   "xla.gpu.reduce_scatter_done");

#undef DEFINE_COLLECTIVE_DONE_OP_LOWERING

template <typename CollectiveIdOp>
class CollectiveIdOpLowering : public OpRewritePattern<CollectiveIdOp> {
  static StringRef Target(ReplicaIdOp) { return "xla.gpu.replica_id"; }
  static StringRef Target(PartitionIdOp) { return "xla.gpu.partition_id"; }

 public:
  CollectiveIdOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern<CollectiveIdOp>(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(CollectiveIdOp op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, Target(op), op);

    // Call the runtime intrinsic with the original operands.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(), TypeRange(),
                                              op->getOperands());
    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

class ReplicaIdOpLowering : public CollectiveIdOpLowering<ReplicaIdOp> {
 public:
  using CollectiveIdOpLowering::CollectiveIdOpLowering;
};

class PartitionIdOpLowering : public CollectiveIdOpLowering<PartitionIdOp> {
 public:
  using CollectiveIdOpLowering::CollectiveIdOpLowering;
};

//===----------------------------------------------------------------------===//
// Host<->Device communication ops lowering (Send/Recv).
//===----------------------------------------------------------------------===//

using lmhlo::RecvDoneOp;
using lmhlo::RecvOp;
using lmhlo::SendDoneOp;
using lmhlo::SendOp;

template <typename OpT, typename Derived>
class HostSendRecvOpLowering : public OpRewritePattern<OpT> {
 public:
  HostSendRecvOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern<OpT>(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter& rewriter) const override {
    if (!op.getIsHostTransfer()) {
      return failure();
    }

    constexpr bool is_done_op =
        is_any<OpT, lmhlo::SendDoneOp, lmhlo::RecvDoneOp>;

    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // For done ops, drop the token input.
    TypeRange input_types =
        is_done_op ? TypeRange() : TypeRange(op->getOperands());
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, Derived::kCustomCallTarget, input_types, TypeRange());

    llvm::SmallVector<NamedAttribute> custom_call_attributes = {
        {b.getStringAttr("channel_handle"), op.getChannelHandleAttr()}};
    if constexpr (!is_done_op) {
      custom_call_attributes.push_back(NamedAttribute(
          b.getStringAttr("frontend_attributes"), op.getFrontendAttributes()));
    }

    // Convert Send/Recv/SendDone/RecvDone to a function call.
    ValueRange inputs =
        is_done_op ? ValueRange() : ValueRange(op->getOperands());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), callee.getName(),
                                              TypeRange(), inputs);
    AppendCustomCallAttrs(call, custom_call_attributes);

    if constexpr (!is_done_op) {
      // For communication operation we need to produce a fake token, that will
      // be later removed, because corresponding `done` operation doesn't have
      // the token argument. We rely on the `unrealized_conversion_cast`
      // operation to create a fake token from the `i8` constant.
      Value token = op.getResult();
      Value c0 = b.create<arith::ConstantOp>(b.getI8IntegerAttr(0));
      auto fake = b.create<UnrealizedConversionCastOp>(token.getType(), c0);
      token.replaceAllUsesWith(fake.getResult(0));
    }

    // Erase the original operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

#define DEFINE_HOST_SENDRECV_OP_LOWERING(OP, custom_call)          \
  struct Host##OP##Lowering                                        \
      : public HostSendRecvOpLowering<OP, Host##OP##Lowering> {    \
    static constexpr const char kCustomCallTarget[] = custom_call; \
    using HostSendRecvOpLowering::HostSendRecvOpLowering;          \
  }

DEFINE_HOST_SENDRECV_OP_LOWERING(SendOp, "xla.gpu.send_host");
DEFINE_HOST_SENDRECV_OP_LOWERING(SendDoneOp, "xla.gpu.send_done_host");
DEFINE_HOST_SENDRECV_OP_LOWERING(RecvOp, "xla.gpu.recv_host");
DEFINE_HOST_SENDRECV_OP_LOWERING(RecvDoneOp, "xla.gpu.recv_done_host");

//===----------------------------------------------------------------------===//

template <typename PairT, typename... Remaining>
static WalkResult AssignAsyncUid(Operation* op,
                                 CollectiveUidGenerator& collective_uid) {
  auto start = dyn_cast<typename PairT::first_type>(op);
  if (!start) {
    if constexpr (sizeof...(Remaining) != 0) {
      return AssignAsyncUid<Remaining...>(op, collective_uid);
    } else {
      return WalkResult::advance();
    }
  }

  Value token = start.getToken();

  // We expect the token to be consumed just once.
  if (!token.hasOneUse()) return start.emitOpError("token has multiple uses");

  // Token must be consumed by the corresponding done operation.
  auto done = dyn_cast<typename PairT::second_type>(*token.getUsers().begin());
  if (!done) return start.emitOpError("illegal token user");

  collective_uid.AssignUid(start, done);
  return WalkResult::advance();
}

void ConvertLmhloToGpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCallDeclarations custom_calls(std::move(sym_table));

  // Convert lmhlo operations to XLA gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<TerminatorOpLowering, CaseOpLowering, WhileOpLowering>(ctx);
  patterns.insert<InfeedOpLowering, OutfeedOpLowering, CustomCallOpLowering>(
      ctx, custom_calls);

  UidGenerator fft_uid;
  patterns.insert<FftOpLowering>(ctx, fft_uid, custom_calls);

  // Assign shared unique id to each unique pair of async start-done operations,
  // all other collective operations will get assigned uid.
  CollectiveUidGenerator collective_uid;
  auto walked = module.walk([&collective_uid](Operation* op) {
    return AssignAsyncUid<
        std::pair<AllGatherStartOp, AllGatherDoneOp>,
        std::pair<AllReduceStartOp, AllReduceDoneOp>,
        std::pair<AllToAllStartOp, AllToAllDoneOp>,
        std::pair<CollectivePermuteStartOp, CollectivePermuteDoneOp>,
        std::pair<ReduceScatterStartOp, ReduceScatterDoneOp>>(op,
                                                              collective_uid);
  });
  if (walked.wasInterrupted()) return signalPassFailure();

  // Convert lmhlo collective operations to XLA gpu runtime custom calls.
  patterns.insert<PartitionIdOpLowering, ReplicaIdOpLowering>(ctx,
                                                              custom_calls);
  patterns.insert<AllGatherOpLowering, AllGatherStartOpLowering,
                  AllReduceOpLowering, AllReduceStartOpLowering,
                  AllToAllOpLowering, AllToAllStartOpLowering,
                  CollectivePermuteOpLowering, CollectivePermuteStartOpLowering,
                  ReduceScatterOpLowering, ReduceScatterStartOpLowering>(
      ctx, collective_uid, custom_calls);

  // Convert lmhlo host<->device point-to-point communication operations to XLA
  // gpu runtime.
  patterns.insert<HostSendOpLowering, HostSendDoneOpLowering,
                  HostRecvOpLowering, HostRecvDoneOpLowering>(ctx,
                                                              custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();

  // TODO(ezhulenev): We must run `done` op lowering after the `start` op
  // lowering to ensure that all redundant collective operations will be
  // safely replaced by a `memcpy` operations.
  //
  // This should be a part of lmhlo operation canonicalization.
  {
    RewritePatternSet patterns(ctx);
    patterns.insert<AllGatherDoneOpLowering, AllReduceDoneOpLowering,
                    AllToAllDoneOpLowering, CollectivePermuteDoneOpLowering,
                    ReduceScatterDoneOpLowering>(ctx, collective_uid,
                                                 custom_calls);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToGpuRuntimePass() {
  return std::make_unique<ConvertLmhloToGpuRuntimePass>();
}

}  // namespace gpu
}  // namespace xla
