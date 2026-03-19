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
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
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

LogicalResult LowerBlockBarrierOp(BlockBarrierOp block_barrier,
                                  PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(block_barrier.getLoc(), rewriter);

  mlir::TypedValue<triton::PointerType> signal_buffers_arg =
      block_barrier.getSignalBuffers();
  const mlir::TypedValue<mlir::IntegerType> rank =
      block_barrier.getDeviceRank();
  const mlir::TypedValue<mlir::IntegerType> signal_value =
      block_barrier.getSignalValue();
  const int32_t world_size = block_barrier.getWorldSize();
  // Triton magic constant.
  constexpr int32_t kGlobalAddressSpace = 1;

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
        // !tt.ptr<i64>
        const auto ptr_to_i64_type = mlir::triton::PointerType::get(
            rewriter.getI64Type(), kGlobalAddressSpace);
        // tensor<world_size x !tt.ptr<i32>>
        const auto tensor_of_ptr_to_i32_type =
            mlir::RankedTensorType::get({world_size}, ptr_to_i32_type);
        // tensor<world_size x !tt.ptr<i64>>
        const auto tensor_of_i64_ptr_type =
            mlir::RankedTensorType::get({world_size}, ptr_to_i64_type);
        // -----
        // Ops (-> implies return type of the op declared in the next line)
        // -----
        // Triton seems to fail to do pointer arithmetic on pointer of
        // pointers. So we cast the inner one to i64.
        // -> !tt.ptr<i64>
        auto signal_buffers_i64 = mlir::triton::BitcastOp::create(
            builder, ptr_to_i64_type, signal_buffers_arg);
        // SignalBuffers[WorldSize][BlockSize][WorldSize]
        // -> tensor<world_size x !tt.ptr<i64>>
        auto signal_buffers_tensor = mlir::triton::SplatOp::create(
            builder, tensor_of_i64_ptr_type, signal_buffers_i64);
        // -> tensor<world_size x i32>
        auto all_ranks = mlir::triton::MakeRangeOp::create(
            builder, i32_tensor_type, 0, world_size);
        // Pointer to SignalBuffers[0..WorldSize]
        // -> tensor<world_size x !tt.ptr<i64>>
        auto signal_buffer_ptr = mlir::triton::AddPtrOp::create(
            builder, tensor_of_i64_ptr_type, signal_buffers_tensor, all_ranks);
        // SignalBuffers[0..WorldSize]
        // -> tensor<world_size x i64>
        auto signal_buffer_i64 = mlir::triton::LoadOp::create(
            builder,
            /*ptr=*/signal_buffer_ptr,
            /*cache=*/mlir::triton::CacheModifier::NONE,
            /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
            /*isVolatile=*/false);
        // -> tensor<world_size x !tt.ptr<i32>>
        auto signal_buffer = mlir::triton::IntToPtrOp::create(
            builder, tensor_of_ptr_to_i32_type, signal_buffer_i64);
        auto block_offset = mlir::arith::MulIOp::create(
            builder, i32_type, block_id, world_size_op);
        auto block_offset_plus_rank =
            mlir::arith::AddIOp::create(builder, i32_type, block_offset, rank);
        // -> tensor<world_size x i32>
        auto block_offset_plus_rank_tensor = mlir::triton::SplatOp::create(
            builder, i32_tensor_type, block_offset_plus_rank);
        // SignalBuffers[0..WorldSize][block_id][rank]
        // -> tensor<world_size x !tt.ptr<i32>>
        auto signal_addresses = mlir::triton::AddPtrOp::create(
            builder, tensor_of_ptr_to_i32_type, signal_buffer,
            block_offset_plus_rank_tensor);
        // Signal all ranks on the same block id.
        mlir::triton::xla::AtomicWriteOp::create(
            builder,
            /*resultTypes=*/mlir::TypeRange{},
            /*ptr=*/signal_addresses,
            /*signal_value=*/signal_value,
            /*mask=*/mlir::Value{},
            /*scope=*/mlir::triton::MemSyncScope::SYSTEM,
            /*sem=*/mlir::triton::MemSemantic::RELEASE);
        // Pointer to SignalBuffers[rank]
        // -> !tt.ptr<i64>
        auto read_address_ptr_to_i64 = mlir::triton::AddPtrOp::create(
            builder, signal_buffers_i64.getType(), signal_buffers_i64, rank);
        // SignalBuffers[rank]
        // -> i64
        auto read_address_i64 = mlir::triton::LoadOp::create(
            builder,
            /*ptr=*/read_address_ptr_to_i64,
            /*cache=*/mlir::triton::CacheModifier::NONE,
            /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
            /*isVolatile=*/false);
        // -> !tt.ptr<i32>
        auto read_address = mlir::triton::IntToPtrOp::create(
            builder, ptr_to_i32_type, read_address_i64);
        // Pointer to SignalBuffers[rank][block_id]
        // -> !tt.ptr<i32>
        auto read_address_at_block_offset = mlir::triton::AddPtrOp::create(
            builder, ptr_to_i32_type, read_address, block_offset);
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
            /*scope=*/mlir::triton::MemSyncScope::SYSTEM,
            /*sem=*/mlir::triton::MemSemantic::ACQUIRE,
            /*comparator=*/Comparator::LT);
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

std::unique_ptr<Pass> CreateTritonXLALowerBlockBarrierPass() {
  return std::make_unique<TritonXLALowerBlockBarrierPass>();
}

}  // namespace mlir::triton::xla
