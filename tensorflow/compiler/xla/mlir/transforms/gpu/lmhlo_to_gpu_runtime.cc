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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/gpu/custom_calls.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::MemcpyOp;

using mlir::lmhlo::CaseOp;
using mlir::lmhlo::CustomCallOp;
using mlir::lmhlo::FftOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;
using mlir::lmhlo::TerminatorOp;
using mlir::lmhlo::WhileOp;

class ConvertLmhloToGpuRuntimePass
    : public ConvertLmhloToGpuRuntimePassBase<ConvertLmhloToGpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<arith::ArithmeticDialect, cf::ControlFlowDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
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
  IoFeedOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
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
  CustomCalls& custom_calls_;
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
  CustomCallOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // By default all operands passed to the custom call handler.
    llvm::SmallVector<Value> operands = op.getOperands();

    // If custom call has target arguments mapping, then we need to pass empty
    // memrefs in place of holes.
    if (op.getTargetArgMapping().has_value()) {
      auto mapping = *op.getTargetArgMapping();
      int64_t num_args = mapping.getNumArgs();
      int64_t num_results = mapping.getNumResults();

      // We represent holes as empty i8 memrefs.
      Value hole =
          b.create<memref::AllocaOp>(MemRefType::get({0}, b.getI8Type()));
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
  CustomCalls& custom_calls_;
};

//===----------------------------------------------------------------------===//

class FftOpLowering : public OpRewritePattern<FftOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.fft";

 public:
  FftOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(FftOp op,
                                PatternRewriter& rewriter) const override {
    // Create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, kCustomCallTarget, op);

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("fft_length"), op.getFftLengthAttr()},
        {b.getStringAttr("fft_type"), op.getFftTypeAttr()}};

    // Convert Fft to a function call.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, custom_call_attrs);
    return success();
  }

 private:
  CustomCalls& custom_calls_;
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

    // TODO(ezhulenev): We need to make sure that `alloca` is placed in the
    // parent function entry block.
    // https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
    Value index_on_host = b.create<memref::AllocaOp>(index_type);
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

//===----------------------------------------------------------------------===//

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
using mlir::lmhlo_gpu::AllReduceDoneOp;
using mlir::lmhlo_gpu::AllReduceStartOp;

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
  static StringRef Target(AllGatherOp) { return "xla.gpu.all_gather"; }
  static StringRef Target(AllReduceOp) { return "xla.gpu.all_reduce"; }
  static StringRef Target(AllToAllOp) { return "xla.gpu.all_to_all"; }
  static StringRef Target(ReduceScatterOp) { return "xla.gpu.reduce_scatter"; }
  static StringRef Target(CollectivePermuteOp) {
    return "xla.gpu.collective_permute";
  }
  static StringRef Target(AllReduceStartOp) {
    return "xla.gpu.all_reduce_start";
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

  static LogicalResult TryDegenerateToMemCopy(
      CollectivePermuteOp op, const NcclCollectiveConfig& config,
      int replica_count, int num_partitions, PatternRewriter& rewriter) {
    if (!NcclCollectivePermuteThunk::IsDegenerate(op, replica_count,
                                                  num_partitions)) {
      return failure();
    }

    rewriter.create<gpu::MemcpyOp>(
        op.getLoc(), TypeRange(),
        ValueRange({op.getOutput(), op.getOperand()}));

    return success();
  }

  static bool CanImplement(AllGatherOp op) {
    return NcclAllGatherThunk::CanImplement(op);
  }

  static bool CanImplement(AllReduceOp op) {
    return NcclAllReduceThunk::CanImplement(op);
  }

  static bool CanImplement(AllReduceStartOp op) {
    return NcclAllReduceStartThunk::CanImplement(op);
  }

  static bool CanImplement(ReduceScatterOp op) {
    return NcclReduceScatterThunk::CanImplement(op);
  }

  static bool CanImplement(AllToAllOp op) {
    return NcclAllToAllThunk::CanImplement(op);
  }

  static bool CanImplement(CollectivePermuteOp op) {
    return NcclCollectivePermuteThunk::CanImplement(op);
  }

  template <typename ReduceOp>
  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b, ReduceOp op,
                                        func::CallOp call) {
    std::optional<xla::ReductionKind> reduction_kind =
        NcclAllReduceThunkBase::MatchAllReduceComputation(op.getComputation());
    if (!reduction_kind.has_value())
      return op.emitOpError()
             << "Failed to determine reduction computation for AllReduce";

    call->setAttr(
        b.getStringAttr("reduction_kind"),
        b.getI64IntegerAttr(static_cast<int64_t>(reduction_kind.value())));

    return success();
  }

  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b, AllGatherOp op,
                                        func::CallOp call) {
    return success();
  }

  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b, AllToAllOp op,
                                        func::CallOp call) {
    call->setAttr(b.getStringAttr("has_split_dimension"),
                  b.getBoolAttr(op.getSplitDimension().has_value()));
    return success();
  }

  static LogicalResult SetSpecificAttrs(ImplicitLocOpBuilder& b,
                                        CollectivePermuteOp op,
                                        func::CallOp call) {
    auto source_target_pairs_or =
        ConvertNx2Attribute(op.getSourceTargetPairs());
    if (!source_target_pairs_or.ok()) {
      return op.emitOpError()
             << source_target_pairs_or.status().error_message();
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

 public:
  CollectiveOpLowering(MLIRContext* ctx, CollectiveUidGenerator& uid,
                       CustomCalls& custom_calls)
      : OpRewritePattern<CollectiveOp>(ctx),
        uid_(uid),
        custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(CollectiveOp op,
                                PatternRewriter& rewriter) const override {
    // Channel ID is not supported.
    if (op.getChannelId().has_value())
      return op.emitOpError() << "Collective channel id is not supported";

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

    if (!CanImplement(op)) {
      return op.emitOpError()
             << "Requested " << Target(op)
             << " not implemented on GPU; replica_count: " << replica_count
             << ", num_partitions: " << num_partitions << ", group_mode: "
             << CollectiveOpGroupModeToString(config.group_mode)
             << ", NCCL support: " << NcclCollectiveThunk::NcclIsEnabled();
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

    // For asynchonous start operation we need to produce a fake token, that
    // will be later removed, because corresponding `done` operation doesn't
    // have the token argument. We rely on the `unrealized_conversion_cast`
    // operation to create a fake token from the `i8` constant.
    if (auto start = dyn_cast<AllReduceStartOp>(op.getOperation())) {
      Value token = start.getToken();
      Value c0 = b.create<arith::ConstantOp>(b.getI8IntegerAttr(0));
      auto fake = b.create<UnrealizedConversionCastOp>(token.getType(), c0);
      token.replaceAllUsesWith(fake.getResult(0));
    }

    // Erase the original collective operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  CollectiveUidGenerator& uid_;
  CustomCalls& custom_calls_;
};

#define DEFINE_COLLECTIVE_OP_LOWERING(OP)                \
  class OP##Lowering : public CollectiveOpLowering<OP> { \
   public:                                               \
    using CollectiveOpLowering::CollectiveOpLowering;    \
  }

DEFINE_COLLECTIVE_OP_LOWERING(AllGatherOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllReduceOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllReduceStartOp);
DEFINE_COLLECTIVE_OP_LOWERING(ReduceScatterOp);
DEFINE_COLLECTIVE_OP_LOWERING(AllToAllOp);
DEFINE_COLLECTIVE_OP_LOWERING(CollectivePermuteOp);

#undef DEFINE_COLLECTIVE_OP_LOWERING

class AllReduceDoneOpLowering : public OpRewritePattern<AllReduceDoneOp> {
  static constexpr const char kCustomCallTarget[] = "xla.gpu.all_reduce_done";

 public:
  AllReduceDoneOpLowering(MLIRContext* ctx, CollectiveUidGenerator& uid,
                          CustomCalls& custom_calls)
      : OpRewritePattern(ctx), uid_(uid), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(AllReduceDoneOp op,
                                PatternRewriter& rewriter) const override {
    // For done operation we drop the token argument and communicate async event
    // dependency through the `uid` attribute.
    llvm::SmallVector<Value> operands = op.getOperands().drop_front();

    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCustomCallTarget, TypeRange(ValueRange(operands)), TypeRange());

    // Get a unique collective operation id.
    FailureOr<int32_t> uid = uid_.AssignedUid(op);
    if (failed(uid))
      return op.emitOpError("failed to get a unique collective operation id");

    llvm::SmallVector<NamedAttribute> custom_call_attributes = {
        {b.getStringAttr("uid"), b.getI32IntegerAttr(*uid)}};

    // Convert AllReduceDone to a function call.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), operands);
    AppendCustomCallAttrs(call, custom_call_attributes);

    return success();
  }

 private:
  CollectiveUidGenerator& uid_;
  CustomCalls& custom_calls_;
};

template <typename CollectiveIdOp>
class CollectiveIdOpLowering : public OpRewritePattern<CollectiveIdOp> {
  static StringRef Target(ReplicaIdOp) { return "xla.gpu.replica_id"; }
  static StringRef Target(PartitionIdOp) { return "xla.gpu.partition_id"; }

 public:
  CollectiveIdOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
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
  CustomCalls& custom_calls_;
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

void ConvertLmhloToGpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCalls custom_calls(std::move(sym_table));

  // Convert lmhlo operations to XLA gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<TerminatorOpLowering, CaseOpLowering, WhileOpLowering>(ctx);
  patterns.insert<InfeedOpLowering, OutfeedOpLowering, CustomCallOpLowering,
                  FftOpLowering>(ctx, custom_calls);

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

  // Convert lmhlo collective operations to XLA gpu runtime custom calls.
  patterns.insert<PartitionIdOpLowering, ReplicaIdOpLowering>(ctx,
                                                              custom_calls);
  patterns.insert<AllGatherOpLowering, AllReduceOpLowering,
                  AllReduceStartOpLowering, AllToAllOpLowering,
                  CollectivePermuteOpLowering, ReduceScatterOpLowering>(
      ctx, collective_uid, custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();

  // TODO(ezhulenev): We must run `done` op lowering after the `start` op
  // lowering to ensure that all redundant collective operations will be
  // safely replaced by a `memcpy` operations.
  //
  // This should be a part of lmhlo operation canonicalization.
  {
    RewritePatternSet patterns(ctx);
    patterns.insert<AllReduceDoneOpLowering>(ctx, collective_uid, custom_calls);
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
