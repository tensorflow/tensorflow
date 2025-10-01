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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERREMOTEACCESSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

LogicalResult LowerGetRankOp(GetRankOp get_rank, PatternRewriter& rewriter) {
  mlir::Value metadata = get_rank.getMetadata();
  auto metadata_type = dyn_cast<PointerType>(metadata.getType());
  if (!metadata_type) {
    return rewriter.notifyMatchFailure(get_rank, "Metadata is not a pointer");
  }

  Type expected_result_type = metadata_type.getPointeeType();
  if (get_rank->getResult(0).getType() != expected_result_type) {
    return rewriter.notifyMatchFailure(
        get_rank, "Call result type must match the pointer's element type");
  }

  // The rank id is stored as a first element under the metadata pointer.
  Value loadOp = rewriter.create<LoadOp>(
      get_rank.getLoc(), expected_result_type, metadata,
      /*mask=*/nullptr, /*other=*/nullptr, /*boundaryCheck=*/nullptr,
      /*padding=*/nullptr,
      CacheModifierAttr::get(get_rank.getContext(), CacheModifier::NONE),
      EvictionPolicyAttr::get(get_rank.getContext(), EvictionPolicy::NORMAL),
      /*isVolatile=*/rewriter.getBoolAttr(false));
  rewriter.replaceOp(get_rank, loadOp);
  return success();
}

// The peer address should be computed as follows:
//
// offset = address - metadata->buffer_root_ptrs[metadata->rank].
// peer_address = metadata->buffer_root_ptrs[peer_id] + offset.
LogicalResult LowerGetPeerPtrOp(GetPeerPtrOp get_peer_ptr,
                                PatternRewriter& rewriter) {
  Value metadata = get_peer_ptr.getMetadata();
  auto metadata_type = dyn_cast<PointerType>(metadata.getType());
  if (!metadata_type) {
    return rewriter.notifyMatchFailure(get_peer_ptr,
                                       "Metadata is not a pointer");
  }

  ImplicitLocOpBuilder builder(get_peer_ptr.getLoc(), rewriter);
  Value address = get_peer_ptr.getAddress();
  Value peer_id = get_peer_ptr.getPeerId();
  MLIRContext* ctx = rewriter.getContext();

  // Pointer type.
  Type type_i64 = rewriter.getI64Type();
  Type result_type = get_peer_ptr.getResult().getType();

  // Size of the pointer in bytes.
  Value pointer_size_bytes_const =
      builder.create<arith::ConstantIntOp>(type_i64, sizeof(int64_t));

  // 1. Load metadata->rank.
  Value current_rank_load_op = builder.create<GetRankOp>(metadata);

  // 2. Load metadata->local_buffer_root_ptrs[metadata->rank].
  Value local_buffers_ptrs_offset = builder.create<arith::ConstantIntOp>(
      type_i64, offsetof(CollectiveKernelMetadata, local_buffer_root_ptrs));

  Value rank_offset =
      builder.create<arith::ExtUIOp>(type_i64, current_rank_load_op);
  Value current_rank_offset_bytes =
      builder.create<arith::MulIOp>(rank_offset, pointer_size_bytes_const);
  Value current_ptr_offset_bytes = builder.create<arith::AddIOp>(
      local_buffers_ptrs_offset, current_rank_offset_bytes);

  Value current_range_address = builder.create<AddPtrOp>(
      metadata.getType(), metadata, current_ptr_offset_bytes);

  Value current_range_address_value = builder.create<LoadOp>(
      type_i64, current_range_address,
      /*mask=*/nullptr, /*other=*/nullptr, /*boundaryCheck=*/nullptr,
      /*padding=*/nullptr, CacheModifierAttr::get(ctx, CacheModifier::NONE),
      EvictionPolicyAttr::get(ctx, EvictionPolicy::NORMAL),
      /*isVolatile=*/rewriter.getBoolAttr(false));

  // 3. Calculate offset =
  //      address - metadata->local_buffer_root_ptrs[metadata->rank].
  Value current_range_address_int =
      builder.create<PtrToIntOp>(type_i64, address);
  Value offsetInt = builder.create<arith::SubIOp>(current_range_address_int,
                                                  current_range_address_value);

  // 4. Load metadata->local_buffer_root_ptrs[peer_id].
  Value peer_index = builder.create<arith::ExtUIOp>(type_i64, peer_id);
  Value peer_index_offset_bytes =
      builder.create<arith::MulIOp>(peer_index, pointer_size_bytes_const);
  Value peer_range_offset_bytes = builder.create<arith::AddIOp>(
      local_buffers_ptrs_offset, peer_index_offset_bytes);
  Value peer_range_address = builder.create<AddPtrOp>(
      metadata.getType(), metadata, peer_range_offset_bytes);

  Value peer_range_address_value = builder.create<LoadOp>(
      type_i64, peer_range_address,
      /*mask=*/nullptr, /*other=*/nullptr, /*boundaryCheck=*/nullptr,
      /*padding=*/nullptr, CacheModifierAttr::get(ctx, CacheModifier::NONE),
      EvictionPolicyAttr::get(ctx, EvictionPolicy::NORMAL),
      /*isVolatile=*/rewriter.getBoolAttr(false));

  // 5. Calculate the result address: peerBasePtr + offset.
  Value result_int =
      builder.create<arith::AddIOp>(peer_range_address_value, offsetInt);
  Value result_address = builder.create<IntToPtrOp>(result_type, result_int);
  rewriter.replaceOp(get_peer_ptr, result_address);
  return success();
}

class TritonXLALowerRemoteAccessPass
    : public impl::TritonXLALowerRemoteAccessPassBase<
          TritonXLALowerRemoteAccessPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerGetRankOp);
    patterns.add(LowerGetPeerPtrOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerRemoteAccessPass() {
  return std::make_unique<TritonXLALowerRemoteAccessPass>();
}

}  // namespace mlir::triton::xla
