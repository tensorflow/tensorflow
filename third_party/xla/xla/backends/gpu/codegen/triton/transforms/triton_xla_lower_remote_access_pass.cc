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

#include <memory>
#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERREMOTEACCESSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {
LogicalResult LowerGetRankOp(GetRankOp get_rank, PatternRewriter& rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(get_rank);

  mlir::Value metadata = get_rank.getMetadata();
  auto metadata_type = dyn_cast<mlir::triton::PointerType>(metadata.getType());
  if (!metadata_type) {
    return rewriter.notifyMatchFailure(get_rank, "Metadata is not a pointer");
  }

  mlir::Type expectedResultType = metadata_type.getPointeeType();
  if (get_rank->getResult(0).getType() != expectedResultType) {
    return rewriter.notifyMatchFailure(
        get_rank, "Call result type must match the pointer's element type");
  }

  // The rank id is stored as a first element under the metadata pointer.
  // The structure of the metadata is defined in
  // `xla::gpu::CollectiveKernelMetadata`.
  mlir::Value loadOp = rewriter.create<mlir::triton::LoadOp>(
      get_rank.getLoc(), expectedResultType, metadata,
      /*mask=*/nullptr, /*other=*/nullptr, /*boundaryCheck=*/nullptr,
      /*padding=*/nullptr,
      mlir::triton::CacheModifierAttr::get(get_rank.getContext(),
                                           mlir::triton::CacheModifier::NONE),
      mlir::triton::EvictionPolicyAttr::get(
          get_rank.getContext(), mlir::triton::EvictionPolicy::NORMAL),
      /*isVolatile=*/rewriter.getBoolAttr(false));
  rewriter.replaceOp(get_rank, loadOp);
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
