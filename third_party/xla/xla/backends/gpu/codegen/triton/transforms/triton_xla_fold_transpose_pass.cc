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
#include <type_traits>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAFOLDTRANSPOSEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

LogicalResult FoldTransposeOfLoad(TransOp op, PatternRewriter& rewriter) {
  auto load = op.getSrc().getDefiningOp<LoadOp>();
  if (!load) {
    return rewriter.notifyMatchFailure(op, "Transpose source is not a load.");
  }
  auto make_ptr = load.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_ptr) {
    return rewriter.notifyMatchFailure(op, "Expected load of make_tensor_ptr.");
  }
  if (load.getMask() || load.getOther()) {
    return rewriter.notifyMatchFailure(op, "Unsupported load.");
  }

  auto apply_order = [&](auto range) {
    llvm::SmallVector<std::decay_t<decltype(*range.begin())>> result;
    result.reserve(op.getOrder().size());
    for (auto dim : op.getOrder()) {
      result.push_back(range[dim]);
    }
    return result;
  };

  auto ptr_type =
      PointerType::get(op.getType(), make_ptr.getType().getAddressSpace());
  auto new_make_ptr = rewriter.create<MakeTensorPtrOp>(
      make_ptr.getLoc(), ptr_type, make_ptr.getBase(),
      apply_order(make_ptr.getShape()), apply_order(make_ptr.getStrides()),
      apply_order(make_ptr.getOffsets()), apply_order(make_ptr.getOrder()));

  llvm::SmallVector<bool> boundary_check_bits(op.getType().getRank());
  for (auto dim : load.getBoundaryCheck()) {
    boundary_check_bits[dim] = true;
  }
  llvm::SmallVector<int32_t> new_boundary_check;
  for (auto [dim, value] : llvm::enumerate(apply_order(boundary_check_bits))) {
    if (value) {
      new_boundary_check.push_back(dim);
    }
  }
  auto new_load = rewriter.create<LoadOp>(
      load.getLoc(), new_make_ptr, new_boundary_check, load.getPadding(),
      load.getCache(), load.getEvict(), load.getIsVolatile());

  rewriter.replaceOp(op, new_load.getResult());
  return success();
}

class TritonXLAFoldTransposePass
    : public impl::TritonXLAFoldTransposePassBase<TritonXLAFoldTransposePass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(FoldTransposeOfLoad);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAFoldTransposePass() {
  return std::make_unique<TritonXLAFoldTransposePass>();
}

}  // namespace mlir::triton::xla
