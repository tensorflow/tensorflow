/* Copyright 2025 The OpenXLA Authors.

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
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERBLOCKBARRIERPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Suppose we have 4 blocks (0, 1, 2, 3), 2 ranks (world_size = 2),
// and signal_stride = 2.
//
// Blocks 1 and 3 work on the same scratch tile and are `signal_stride` blocks
// apart:
//   - Block 1 has signal_slot = 0 (first block in the pair).
//   - Block 3 has signal_slot = 1 (second block in the pair).
//
// 1. Find `base_block`, the first block in the pair (block 1):
//    `slot_offset = signal_slot * signal_stride` is how many blocks
//    this block is ahead of `base_block`.
//    Subtracting it gives `base_block = block_id - slot_offset`:
//      - Block 1: base_block = 1 - (0 * 2) = 1
//      - Block 3: base_block = 3 - (1 * 2) = 1
//
// 2. Find `target_block` (`base_block + rank * signal_stride`).
mlir::Value ComputeTargetBlock(mlir::ImplicitLocOpBuilder& builder,
                               mlir::Value block_id, mlir::Value rank,
                               mlir::Value signal_slot, int32_t signal_stride) {
  const mlir::IntegerType i32_type = builder.getI32Type();
  auto stride_op = mlir::arith::ConstantOp::create(
      builder, builder.getI32IntegerAttr(signal_stride));
  auto slot_offset =
      mlir::arith::MulIOp::create(builder, i32_type, signal_slot, stride_op);
  auto base_block =
      mlir::arith::SubIOp::create(builder, i32_type, block_id, slot_offset);
  auto rank_offset =
      mlir::arith::MulIOp::create(builder, i32_type, rank, stride_op);
  return mlir::arith::AddIOp::create(builder, i32_type, base_block,
                                     rank_offset);
}

// Returns a pointer (or 1D tensor of pointers) to `&SignalBuffers[x][y]`.
// `x` and `y` may be scalar `i32` values or 1D `i32` tensors. If `x` is a 1D
// tensor and `y` is a scalar `i32`, `y` is splatted to match `x`'s shape.
mlir::Value GetSignalBufferElementPtr(mlir::ImplicitLocOpBuilder& builder,
                                      mlir::Value signal_buffers_arg,
                                      mlir::Value x, mlir::Value y) {
  constexpr auto kGlobalAddressSpace = mlir::triton::PtrAddrSpace::Global;
  const mlir::IntegerType i32_type = builder.getI32Type();
  const mlir::IntegerType i64_type = builder.getI64Type();
  const auto ptr_to_i32_type =
      mlir::triton::PointerType::get(i32_type, kGlobalAddressSpace);
  const auto ptr_to_i64_type =
      mlir::triton::PointerType::get(i64_type, kGlobalAddressSpace);

  // Triton fails to do pointer arithmetic on pointer of pointers, so we cast
  // the inner pointer to i64.
  mlir::Value signal_buffers_i64 = signal_buffers_arg;
  if (signal_buffers_arg.getType() != ptr_to_i64_type) {
    signal_buffers_i64 = mlir::triton::BitcastOp::create(
        builder, ptr_to_i64_type, signal_buffers_arg);
  }

  mlir::Type ptr_i64_type = ptr_to_i64_type;
  mlir::Type ptr_i32_type = ptr_to_i32_type;
  if (auto ranked_x = mlir::dyn_cast<mlir::RankedTensorType>(x.getType())) {
    ptr_i64_type =
        mlir::RankedTensorType::get(ranked_x.getShape(), ptr_to_i64_type);
    ptr_i32_type =
        mlir::RankedTensorType::get(ranked_x.getShape(), ptr_to_i32_type);
    signal_buffers_i64 = mlir::triton::SplatOp::create(builder, ptr_i64_type,
                                                       signal_buffers_i64);
    if (!mlir::isa<mlir::RankedTensorType>(y.getType())) {
      y = mlir::triton::SplatOp::create(builder, ranked_x, y);
    }
  }

  auto buf_ptr_i64 = mlir::triton::AddPtrOp::create(builder, ptr_i64_type,
                                                    signal_buffers_i64, x);
  auto buf_i64 = mlir::triton::LoadOp::create(builder, /*ptr=*/buf_ptr_i64,
                                              /*isVolatile=*/false);
  auto buf = mlir::triton::IntToPtrOp::create(builder, ptr_i32_type, buf_i64);
  return mlir::triton::AddPtrOp::create(builder, ptr_i32_type, buf, y);
}

// Atomically writes `signal_value` to `SignalBuffers[x][y]`.
void WriteToSignalBuffer(mlir::ImplicitLocOpBuilder& builder,
                         mlir::Value signal_buffers_arg, mlir::Value x,
                         mlir::Value y, mlir::Value signal_value,
                         mlir::triton::MemSyncScope mem_sync_scope,
                         mlir::triton::MemSemantic mem_sync_semantic) {
  mlir::Value signal_addr =
      GetSignalBufferElementPtr(builder, signal_buffers_arg, x, y);
  mlir::triton::xla::AtomicWriteOp::create(
      builder,
      /*resultTypes=*/mlir::TypeRange{},
      /*ptr=*/signal_addr,
      /*value=*/signal_value,
      /*mask=*/mlir::Value{},
      /*mem_sync_scope=*/mem_sync_scope,
      /*mem_sync_semantic=*/mem_sync_semantic);
}

// PairwiseSymmetric barrier mode is lowered separately because it synchronizes
// with a single peer rank using scalar ops on thread 0, whereas the other
// modes synchronize across all world_size ranks using tensor ops on
// world_size threads.
//
// Two atomic writes are needed because unlike other modes, which are
// broadcasting to every rank's buffer and updating the local rank's own
// invocation counter as part of the same write, PairwiseSymmetric
// only signals one peer's buffer and needs a separate local write to advance
// its own invocation counter.
//
// Because only the local and target peer slots are updated, the signal buffer
// is left with different values across slots and must not be shared with
// other operations or barrier modes.
LogicalResult LowerPairwiseSymmetricBlockBarrierOp(BlockBarrierOp block_barrier,
                                                   PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(block_barrier.getLoc(), rewriter);

  mlir::TypedValue<triton::PointerType> signal_buffers_arg =
      block_barrier.getSignalBuffers();
  const mlir::TypedValue<mlir::IntegerType> rank =
      block_barrier.getDeviceRank();
  const mlir::TypedValue<mlir::IntegerType> signal_value =
      block_barrier.getSignalValue();
  const mlir::Value signal_slot = block_barrier.getSignalSlot();
  const int32_t world_size = block_barrier.getWorldSize();
  const int32_t signal_stride = block_barrier.getSignalStride();

  const mlir::TypedValue<mlir::Type> world_size_op =
      mlir::arith::ConstantOp::create(builder,
                                      builder.getI32IntegerAttr(world_size));
  const mlir::TypedValue<mlir::IntegerType> thread_id =
      triton::xla::GetTidOp::create(builder);
  const mlir::TypedValue<mlir::IntegerType> block_id =
      triton::GetProgramIdOp::create(builder, 0);
  auto zero_op =
      mlir::arith::ConstantOp::create(builder, builder.getI32IntegerAttr(0));
  auto tid_is_zero = mlir::arith::CmpIOp::create(
      builder, mlir::arith::CmpIPredicate::eq, thread_id, zero_op);
  mlir::scf::IfOp::create(
      builder,
      /*cond=*/tid_is_zero,
      /*thenBuilder=*/
      [&](mlir::OpBuilder& op_builder, mlir::Location location) {
        mlir::ImplicitLocOpBuilder builder(location, op_builder);
        const mlir::IntegerType i32_type = builder.getI32Type();

        // Update local invocation counter slot
        // SignalBuffers[rank][block_id * world_size + rank].
        auto block_offset = mlir::arith::MulIOp::create(
            builder, i32_type, block_id, world_size_op);
        auto local_counter_offset =
            mlir::arith::AddIOp::create(builder, i32_type, block_offset, rank);
        WriteToSignalBuffer(builder, signal_buffers_arg, rank,
                            local_counter_offset, signal_value,
                            mlir::triton::MemSyncScope::GPU,
                            mlir::triton::MemSemantic::RELAXED);

        // Signal peer at
        // SignalBuffers[signal_slot][target_block * world_size + rank].
        mlir::Value target_block = ComputeTargetBlock(
            builder, block_id, rank, signal_slot, signal_stride);
        auto target_block_offset = mlir::arith::MulIOp::create(
            builder, i32_type, target_block, world_size_op);
        auto peer_write_offset = mlir::arith::AddIOp::create(
            builder, i32_type, target_block_offset, rank);
        WriteToSignalBuffer(builder, signal_buffers_arg, signal_slot,
                            peer_write_offset, signal_value,
                            mlir::triton::MemSyncScope::SYSTEM,
                            mlir::triton::MemSemantic::RELEASE);

        // Wait on SignalBuffers[rank][block_id * world_size + signal_slot].
        auto wait_offset = mlir::arith::AddIOp::create(
            builder, i32_type, block_offset, signal_slot);
        mlir::Value wait_addr = GetSignalBufferElementPtr(
            builder, signal_buffers_arg, rank, wait_offset);
        mlir::triton::xla::AtomicSpinWaitOp::create(
            builder,
            /*resultTypes=*/mlir::TypeRange{},
            /*ptr=*/wait_addr,
            /*expected=*/signal_value,
            /*mask=*/mlir::Value{},
            /*mem_sync_scope=*/mlir::triton::MemSyncScope::SYSTEM,
            /*mem_sync_semantic=*/mlir::triton::MemSemantic::ACQUIRE,
            /*comparator=*/Comparator::GE);
        mlir::scf::YieldOp::create(builder);
      });
  mlir::triton::gpu::BarrierOp::create(builder, triton::gpu::AddrSpace::Local);
  rewriter.eraseOp(block_barrier);
  return success();
}

LogicalResult LowerBlockBarrierOp(BlockBarrierOp block_barrier,
                                  PatternRewriter& rewriter) {
  const BarrierMode barrier_mode = block_barrier.getBarrierMode();
  if (barrier_mode == BarrierMode::kPairwiseSymmetric) {
    return LowerPairwiseSymmetricBlockBarrierOp(block_barrier, rewriter);
  }

  mlir::ImplicitLocOpBuilder builder(block_barrier.getLoc(), rewriter);

  mlir::TypedValue<triton::PointerType> signal_buffers_arg =
      block_barrier.getSignalBuffers();
  const mlir::TypedValue<mlir::IntegerType> rank =
      block_barrier.getDeviceRank();
  const mlir::TypedValue<mlir::IntegerType> signal_value =
      block_barrier.getSignalValue();
  const int32_t world_size = block_barrier.getWorldSize();
  const int32_t signal_stride = block_barrier.getSignalStride();
  // Triton magic constant.
  constexpr auto kGlobalAddressSpace = mlir::triton::PtrAddrSpace::Global;

  const mlir::TypedValue<mlir::Type> world_size_op =
      mlir::arith::ConstantOp::create(builder,
                                      builder.getI32IntegerAttr(world_size));
  const mlir::TypedValue<mlir::IntegerType> thread_id =
      triton::xla::GetTidOp::create(builder);
  const mlir::TypedValue<mlir::IntegerType> block_id =
      triton::GetProgramIdOp::create(builder, 0);

  auto tid_is_lt_world_size = mlir::arith::CmpIOp::create(
      builder, mlir::arith::CmpIPredicate::ult, thread_id, world_size_op);

  // Only the first `world_size` threads will execute this block.
  mlir::scf::IfOp::create(
      builder,
      /*cond=*/tid_is_lt_world_size,
      // Inside if block so tid must be less than world_size.
      /*thenBuilder=*/
      [&](mlir::OpBuilder& op_builder, mlir::Location location) {
        mlir::ImplicitLocOpBuilder builder(location, op_builder);
        // -----
        // Types
        // -----
        const mlir::IntegerType i32_type = builder.getI32Type();
        // tensor<world_size x i32>
        const auto i32_tensor_type =
            mlir::RankedTensorType::get({world_size}, i32_type);
        // !tt.ptr<i32>
        const auto ptr_to_i32_type = mlir::triton::PointerType::get(
            rewriter.getI32Type(), kGlobalAddressSpace);
        // tensor<world_size x !tt.ptr<i32>>
        const auto tensor_of_ptr_to_i32_type =
            mlir::RankedTensorType::get({world_size}, ptr_to_i32_type);
        // -----
        // Ops (-> implies return type of the op declared in the next line)
        // -----
        auto block_offset = mlir::arith::MulIOp::create(
            builder, i32_type, block_id, world_size_op);
        mlir::Value write_offset;
        mlir::Value wait_block_offset = block_offset;
        if (barrier_mode != BarrierMode::kSymmetric) {
          // - In `kConsumerSymmetric`, `target_block` is the block on peer
          //   GPUs that reads this rank's data, so we write to `signal_slot`
          //   inside `target_block`'s slots
          //   (`target_block * world_size + signal_slot`).
          // - In `kProducerSymmetric`, `target_block` is the producer block
          //   whose slots (`target_block * world_size + [0..world_size)`)
          //   this consumer block waits on.
          mlir::Value signal_slot = block_barrier.getSignalSlot();
          mlir::Value target_block = ComputeTargetBlock(
              builder, block_id, rank, signal_slot, signal_stride);
          auto target_block_offset = mlir::arith::MulIOp::create(
              builder, i32_type, target_block, world_size_op);
          if (barrier_mode == BarrierMode::kConsumerSymmetric) {
            write_offset = mlir::arith::AddIOp::create(
                builder, i32_type, target_block_offset, signal_slot);
          } else {
            write_offset = mlir::arith::AddIOp::create(builder, i32_type,
                                                       block_offset, rank);
            wait_block_offset = target_block_offset;
          }
        } else {
          write_offset = mlir::arith::AddIOp::create(builder, i32_type,
                                                     block_offset, rank);
        }
        // -> tensor<world_size x i32>
        auto all_ranks = mlir::triton::MakeRangeOp::create(
            builder, i32_tensor_type, 0, world_size);
        // Signal all ranks at SignalBuffers[0..WorldSize][write_offset].
        WriteToSignalBuffer(builder, signal_buffers_arg, all_ranks,
                            write_offset, signal_value,
                            mlir::triton::MemSyncScope::SYSTEM,
                            mlir::triton::MemSemantic::RELEASE);
        // Pointer to SignalBuffers[rank][wait_block_offset]
        // -> !tt.ptr<i32>
        mlir::Value read_address_at_block_offset = GetSignalBufferElementPtr(
            builder, signal_buffers_arg, rank, wait_block_offset);
        // -> tensor<world_size x !tt.ptr<i32>>
        auto read_address_at_block_offset_tensor =
            mlir::triton::SplatOp::create(builder, tensor_of_ptr_to_i32_type,
                                          read_address_at_block_offset);
        // SignalBuffers[rank][block_id][0..WorldSize]
        // -> tensor<world_size x !tt.ptr<i32>>
        auto wait_addresses = mlir::triton::AddPtrOp::create(
            builder, tensor_of_ptr_to_i32_type,
            read_address_at_block_offset_tensor, all_ranks);
        // Wait for all ranks on the same block id to signal.
        mlir::triton::xla::AtomicSpinWaitOp::create(
            builder,
            /*resultTypes=*/mlir::TypeRange{},
            /*ptr=*/wait_addresses,
            /*expected=*/signal_value,
            /*mask=*/mlir::Value{},
            /*mem_sync_scope=*/mlir::triton::MemSyncScope::SYSTEM,
            /*mem_sync_semantic=*/mlir::triton::MemSemantic::ACQUIRE,
            /*comparator=*/Comparator::GE);
        // Terminate the block.
        mlir::scf::YieldOp::create(builder);
      });
  builder.create<mlir::triton::gpu::BarrierOp>(triton::gpu::AddrSpace::Local);
  rewriter.eraseOp(block_barrier);
  return success();
}

class TritonXLALowerBlockBarrierPass
    : public impl::TritonXLALowerBlockBarrierPassBase<
          TritonXLALowerBlockBarrierPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerBlockBarrierOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla
