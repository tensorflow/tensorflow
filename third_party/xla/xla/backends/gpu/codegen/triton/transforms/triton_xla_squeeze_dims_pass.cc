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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLASQUEEZEDIMSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Returns list of dimensions with size equal to 1. If all the dimensions have
// size 1, do not return the last dimension to avoid tripping up Triton.
SmallVector<uint32_t> GetDimsToSqueeze(RankedTensorType type) {
  SmallVector<uint32_t> result;
  for (auto [dim, size] : llvm::enumerate(type.getShape())) {
    if (size == 1) {
      result.push_back(dim);
    }
  }
  if (result.size() == type.getRank()) {
    result.pop_back();  // Keep one unit dimension.
  }
  return result;
}

// Returns the axis of first squeeze_dims user.
std::optional<uint32_t> GetSqueezeDimsUserAxis(Operation* op) {
  for (Operation* user : op->getUsers()) {
    if (auto op = dyn_cast<SqueezeDimsOp>(user); op) {
      return op.getAxis();
    }
  }
  return std::nullopt;
}

// Replaces 'op' with 'values', which is the op result squeezed along 'axis'.
void ReplaceOpWithExpandDimsOf(PatternRewriter& rewriter, Operation* op,
                               ValueRange values, uint32_t axis) {
  for (auto [result, value] : llvm::zip_equal(op->getResults(), values)) {
    // Replace all squeeze_dims users with the new value.
    for (Operation* user : make_early_inc_range(result.getUsers())) {
      if (auto op = dyn_cast<SqueezeDimsOp>(user); op && op.getAxis() == axis) {
        rewriter.replaceOp(user, value);
      }
    }
    // If any users remain, replace the op with expand_dims.
    if (!result.use_empty()) {
      Value expand_dims = rewriter.create<ExpandDimsOp>(
          op->getLoc(), result.getType(), value, axis);
      rewriter.replaceAllUsesWith(result, expand_dims);
    }
  }
}

// Sets the insertion point at the given op and returns the guard.
[[nodiscard]] OpBuilder::InsertionGuard SetInsertionPoint(OpBuilder& builder,
                                                          Operation* op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  return guard;
}

// Returns a new container with the given dimensions removed.
template <typename ContainerT>
auto SqueezeElements(ContainerT elements, ArrayRef<uint32_t> squeeze_dims) {
  CHECK(absl::c_is_sorted(squeeze_dims));
  auto it = elements.begin();
  SmallVector<typename std::iterator_traits<decltype(it)>::value_type> result;
  for (uint32_t dim : squeeze_dims) {
    CHECK_LT(dim, elements.size());
    auto end = elements.begin() + dim;
    std::copy(it, end, std::back_inserter(result));
    it = std::next(end);
  }
  std::copy(it, elements.end(), std::back_inserter(result));
  return result;
}

// Returns a new boundary check with the given dimensions removed.
SmallVector<int32_t> SqueezeBoundaryCheck(ArrayRef<int32_t> boundary_check,
                                          ArrayRef<uint32_t> squeeze_dims) {
  CHECK(absl::c_is_sorted(boundary_check));
  CHECK(absl::c_is_sorted(squeeze_dims));
  SmallVector<int32_t> result;
  auto it = squeeze_dims.begin();
  for (int32_t dim : boundary_check) {
    it = std::lower_bound(it, squeeze_dims.end(), dim);
    if (it != squeeze_dims.end() && *it == dim) {
      continue;
    }
    result.push_back(dim - (it - squeeze_dims.begin()));
  }
  return result;
}

// Returns a new tensor type with the given dimensions removed.
RankedTensorType SqueezeTensorType(RankedTensorType type,
                                   ArrayRef<uint32_t> squeeze_dims) {
  SmallVector<int64_t> shape = SqueezeElements(type.getShape(), squeeze_dims);
  Attribute encoding = type.getEncoding();
  if (encoding) {
    auto inferLayoutInterface =
        cast<DialectInferLayoutInterface>(&encoding.getDialect());
    [[maybe_unused]] LogicalResult result =
        inferLayoutInterface->inferReshapeOpEncoding(
            type.getShape(), encoding, shape, encoding, std::nullopt);
    CHECK(succeeded(result));
  }
  return RankedTensorType::get(shape, type.getElementType(), encoding);
}

// Returns a new tensor value with the given dimensions removed.
// Low dimensions are squeezed first.
Value SqueezeTensorValue(PatternRewriter& rewriter, Value value,
                         ArrayRef<uint32_t> squeeze_dims) {
  CHECK(absl::c_is_sorted(squeeze_dims));
  for (uint32_t i = 0; i < squeeze_dims.size(); ++i) {
    uint32_t dim = squeeze_dims[i] - i;
    Type type = SqueezeTensorType(cast<RankedTensorType>(value.getType()), dim);
    value = rewriter.create<SqueezeDimsOp>(value.getLoc(), type, value, dim);
  }
  return value;
}

// Returns a new pointer by applying the given offsets and strides for the
// given dimensions.
Value SqueezePointer(PatternRewriter& rewriter, Location loc, Value base,
                     ValueRange offsets, ValueRange strides,
                     ArrayRef<uint32_t> squeeze_dims) {
  for (auto dim : squeeze_dims) {
    Value extsi = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(),
                                                  offsets[dim]);
    Value muli = rewriter.create<arith::MulIOp>(loc, extsi, strides[dim]);
    base = rewriter.create<AddPtrOp>(loc, base.getType(), base, muli);
  }
  return base;
}

// Rewrites tt.make_tensor_ptr with unit dimensions. Returns the
// new MakeTensorPtrOp result and the dimensions that were removed.
Value SqueezeMakeTensorPtr(PatternRewriter& rewriter, MakeTensorPtrOp op,
                           ArrayRef<uint32_t> squeeze_dims) {
  auto tensor_type = cast<RankedTensorType>(op.getType().getPointeeType());
  auto squeeze_type = SqueezeTensorType(tensor_type, squeeze_dims);
  auto ptr_type =
      PointerType::get(squeeze_type, op.getType().getAddressSpace());

  // Strides already encode the layout, so we can use the default order.
  // Note that the order attribute is ignored in the Triton lowering.
  SmallVector<int32_t> order(squeeze_type.getShape().size());
  std::iota(order.rbegin(), order.rend(), 0);

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, op);
  // Add the offsets along the dimensions to squeeze to the base pointer.
  Value base = SqueezePointer(rewriter, op.getLoc(), op.getBase(),
                              op.getOffsets(), op.getStrides(), squeeze_dims);
  return rewriter.create<MakeTensorPtrOp>(
      op.getLoc(), ptr_type, base, SqueezeElements(op.getShape(), squeeze_dims),
      SqueezeElements(op.getStrides(), squeeze_dims),
      SqueezeElements(op.getOffsets(), squeeze_dims), order);
}

// Folds squeeze_dims into tt.load(tt.make_tensor_ptr).
// TODO(csigg): Add support for tt.load(tt.make_tensor_descriptor).
LogicalResult FoldSqueezeDimsOfLoad(LoadOp op, PatternRewriter& rewriter) {
  if (op.getMask() || op.getOther()) {
    return rewriter.notifyMatchFailure(op, "Unsupported load.");
  }
  std::optional<uint32_t> axis = GetSqueezeDimsUserAxis(op);
  if (!axis) {
    return rewriter.notifyMatchFailure(op, "No squeeze_dims users.");
  }
  if (absl::c_contains(op.getBoundaryCheck(), *axis)) {
    return rewriter.notifyMatchFailure(op, "Boundary check contains axis.");
  }
  auto make_tensor_ptr = op.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_tensor_ptr) {
    return rewriter.notifyMatchFailure(
        op, "Expected ptr to be defined by make_tensor_ptr.");
  }

  Value pointer = SqueezeMakeTensorPtr(rewriter, make_tensor_ptr, *axis);
  Value new_load = rewriter.create<LoadOp>(
      op.getLoc(), pointer, SqueezeBoundaryCheck(op.getBoundaryCheck(), *axis),
      op.getPadding(), op.getCache(), op.getEvict(), op.getIsVolatile());
  ReplaceOpWithExpandDimsOf(rewriter, op, new_load, *axis);
  return success();
}

// Extracts unit dimensions from tt.store and prepends them as squeeze_dims.
LogicalResult SqueezeStore(StoreOp op, PatternRewriter& rewriter) {
  if (op.getMask()) {
    return rewriter.notifyMatchFailure(op, "Unsupported store.");
  }
  auto make_tensor_ptr = op.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_tensor_ptr) {
    return rewriter.notifyMatchFailure(
        op, "Expected ptr to be defined by make_tensor_ptr.");
  }
  auto tensor_type = dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!tensor_type || tensor_type.getRank() == 0) {
    return rewriter.notifyMatchFailure(op, "Expected tensor type.");
  }

  auto squeeze_dims = GetDimsToSqueeze(tensor_type);
  if (squeeze_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  Value pointer = SqueezeMakeTensorPtr(rewriter, make_tensor_ptr, squeeze_dims);
  Value value = SqueezeTensorValue(rewriter, op.getValue(), squeeze_dims);
  rewriter.replaceOpWithNewOp<StoreOp>(
      op, pointer, value,
      SqueezeBoundaryCheck(op.getBoundaryCheck(), squeeze_dims), op.getCache(),
      op.getEvict());
  return success();
}

// Extracts unit dimensions from the tt.reshape operand and prepends them as
// squeeze_dims.
LogicalResult SqueezeReshapeOperand(ReshapeOp op, PatternRewriter& rewriter) {
  if (op.getAllowReorderAttr() || op.getEfficientLayoutAttr()) {
    return rewriter.notifyMatchFailure(op, "Unsupported reshape.");
  }
  auto squeeze_dims = GetDimsToSqueeze(op.getSrc().getType());
  if (squeeze_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  Value value = SqueezeTensorValue(rewriter, op.getSrc(), squeeze_dims);
  rewriter.modifyOpInPlace(op, [&]() { op.setOperand(value); });
  return success();
}

// Extracts unit dimensions from the tt.reshape result and appends them as
// expand_dims.
LogicalResult ExpandReshapeResult(ReshapeOp op, PatternRewriter& rewriter) {
  if (op.getAllowReorderAttr() || op.getEfficientLayoutAttr()) {
    return rewriter.notifyMatchFailure(op, "Unsupported reshape.");
  }
  auto expand_dims = GetDimsToSqueeze(op.getType());
  if (expand_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  Value result = rewriter.create<ReshapeOp>(
      op.getLoc(), SqueezeTensorType(op.getType(), expand_dims), op.getSrc());
  for (int32_t i = expand_dims.size() - 1; i >= 0; --i) {
    uint32_t dim = expand_dims[i] - i;
    result = rewriter.create<ExpandDimsOp>(op.getLoc(), result, dim);
  }
  rewriter.replaceOp(op, result);
  return success();
}

// Pushes squeeze_dims up through element-wise operations.
// Example:
//   %1 = elementwise-op %0
//   %2 = squeeze_dims %1
// is rewritten to:
//   %1 = squeeze_dims %0
//   %2 = elementwise-op %1
class PushSqueezeDimsUpThroughElementwise final
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
 public:
  using OpTraitRewritePattern::OpTraitRewritePattern;

 private:
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    std::optional<uint32_t> axis = GetSqueezeDimsUserAxis(op);
    if (!axis) {
      return rewriter.notifyMatchFailure(op, "No squeeze_dims users.");
    }

    OperationState state(op->getLoc(), op->getName());
    for (Type type : op->getResultTypes()) {
      state.addTypes(SqueezeTensorType(cast<RankedTensorType>(type), *axis));
    }
    for (Value operand : op->getOperands()) {
      if (isa<RankedTensorType>(operand.getType())) {
        operand = SqueezeTensorValue(rewriter, operand, *axis);
      }
      state.addOperands(operand);
    }
    state.addAttributes(op->getAttrs());
    Operation* new_op = rewriter.create(state);
    ReplaceOpWithExpandDimsOf(rewriter, op, new_op->getResults(), *axis);
    return success();
  }
};

// Pushes squeeze_dims up through tt.broadcast.
// Example:
//   %1 = tt.broadcast %0
//   %2 = squeeze_dims %1
// is rewritten to:
//   %1 = squeeze_dims %0
//   %2 = tt.broadcast %1
LogicalResult PushSqueezeDimsUpThroughBroadcast(SqueezeDimsOp op,
                                                PatternRewriter& rewriter) {
  auto broadcast = op.getSrc().getDefiningOp<BroadcastOp>();
  if (!broadcast) {
    return rewriter.notifyMatchFailure(op, "Expected broadcast producer.");
  }

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, broadcast);
  Value value = SqueezeTensorValue(rewriter, broadcast.getSrc(), op.getAxis());
  Value new_broadcast =
      rewriter.create<BroadcastOp>(broadcast.getLoc(), op.getType(), value);
  ReplaceOpWithExpandDimsOf(rewriter, broadcast, new_broadcast, op.getAxis());
  return success();
}

// Pushes squeeze_dims up through tt.trans.
// Example:
//   %1 = tt.trans %0, perm = [1, 0]
//   %2 = squeeze_dims %1, axis = 0
// is rewritten to:
//   %1 = squeeze_dims %0, axis = 1
//   %2 = tt.trans %1, perm = [0]
LogicalResult PushSqueezeDimsUpThroughTrans(SqueezeDimsOp op,
                                            PatternRewriter& rewriter) {
  auto trans = op.getSrc().getDefiningOp<TransOp>();
  if (!trans) {
    return rewriter.notifyMatchFailure(op, "Expected trans producer.");
  }

  auto order = trans.getOrder();
  uint32_t dst_axis = op.getAxis();
  uint32_t src_axis = order[dst_axis];

  // Compute the new permutation for the transpose.
  auto new_order = SqueezeElements(order, dst_axis);
  for (auto& dim : new_order) {
    dim -= dim > src_axis;
  }

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, trans);
  Value value = SqueezeTensorValue(rewriter, trans.getSrc(), src_axis);
  Value new_trans = rewriter.create<TransOp>(trans.getLoc(), value, new_order);
  ReplaceOpWithExpandDimsOf(rewriter, trans, new_trans, dst_axis);
  return success();
}

// Pushes squeeze_dims up through tt.join.
// Example:
//   %2 = tt.join %0, %1
//   %3 = squeeze_dims %2
// is rewritten to:
//   %2 = squeeze_dims %0
//   %3 = squeeze_dims %1
//   %4 = tt.join %2, %3
LogicalResult PushSqueezeDimsUpThroughJoin(SqueezeDimsOp op,
                                           PatternRewriter& rewriter) {
  auto join = op.getSrc().getDefiningOp<JoinOp>();
  if (!join) {
    return rewriter.notifyMatchFailure(op, "Expected join producer.");
  }

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, join);
  SmallVector<Value> operands;
  operands.reserve(join.getOperands().size());
  for (Value operand : join.getOperands()) {
    operands.push_back(SqueezeTensorValue(rewriter, operand, op.getAxis()));
  }

  Value new_join =
      rewriter.create<JoinOp>(join.getLoc(), op.getType(), operands);
  ReplaceOpWithExpandDimsOf(rewriter, join, new_join, op.getAxis());
  return success();
}

// Pushes squeeze_dims up through tt.reduce.
// Example:
//   %1 = tt.reduce(%0) axis=1
//   %2 = squeeze_dims %1, axis=0
// is rewritten to:
//   %1 = squeeze_dims %0, axis=0
//   %2 = tt.reduce(%1) axis=0
LogicalResult PushSqueezeDimsUpThroughReduce(SqueezeDimsOp op,
                                             PatternRewriter& rewriter) {
  auto reduce = op.getSrc().getDefiningOp<ReduceOp>();
  if (!reduce || reduce.getNumResults() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Expected single-result reduce producer.");
  }

  uint32_t squeeze_axis = op.getAxis() + (op.getAxis() >= reduce.getAxis());
  uint32_t reduce_axis = reduce.getAxis() - (op.getAxis() < reduce.getAxis());

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, reduce);
  SmallVector<Value> operands;
  operands.reserve(reduce.getOperands().size());
  for (Value operand : reduce.getOperands()) {
    operands.push_back(SqueezeTensorValue(rewriter, operand, squeeze_axis));
  }

  auto new_reduce = rewriter.create<ReduceOp>(reduce.getLoc(), op.getType(),
                                              operands, reduce_axis);
  rewriter.cloneRegionBefore(reduce->getRegion(0), new_reduce->getRegion(0),
                             new_reduce->getRegion(0).begin());
  ReplaceOpWithExpandDimsOf(rewriter, reduce, new_reduce->getResult(0),
                            op.getAxis());
  return success();
}

// Pushes squeeze_dims up through tt.expand_dims, or folds them.
// Example:
//   %1 = tt.expand_dims %0, axis=1
//   %2 = squeeze_dims %1, axis=0
// is rewritten to:
//   %1 = squeeze_dims %0, axis=0
//   %2 = tt.expand_dims %1, axis=0
LogicalResult PushSqueezeDimsUpThroughExpandDims(SqueezeDimsOp op,
                                                 PatternRewriter& rewriter) {
  auto expand_dims = op.getSrc().getDefiningOp<ExpandDimsOp>();
  if (!expand_dims) {
    return rewriter.notifyMatchFailure(op, "Expected expand_dims producer.");
  }

  if (expand_dims.getSrc().getType() == op.getType()) {
    rewriter.replaceOp(op, expand_dims.getSrc());
    return success();
  }

  // Swap: squeeze_dims(expand_dims) -> expand_dims(squeeze_dims)
  uint32_t expand_axis = expand_dims.getAxis();
  uint32_t squeeze_axis = op.getAxis();
  CHECK_NE(expand_axis, squeeze_axis);

  squeeze_axis -= squeeze_axis > expand_axis;
  expand_axis -= expand_axis > squeeze_axis;

  OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, expand_dims);
  Value value =
      SqueezeTensorValue(rewriter, expand_dims.getSrc(), squeeze_axis);
  rewriter.replaceOpWithNewOp<ExpandDimsOp>(op, op.getType(), value,
                                            expand_axis);
  return success();
}

// Reorders squeeze_dims ops to enforce the invariant that lower-axis ops
// come first.
// Example:
//   %1 = squeeze_dims %0, axis = 2
//   %2 = squeeze_dims %1, axis = 0
// is rewritten to:
//   %1 = squeeze_dims %0, axis = 0
//   %2 = squeeze_dims %1, axis = 1
LogicalResult ReorderSqueezeDims(SqueezeDimsOp op, PatternRewriter& rewriter) {
  auto inner = op.getSrc().getDefiningOp<SqueezeDimsOp>();
  if (!inner || inner.getAxis() <= op.getAxis()) {
    return rewriter.notifyMatchFailure(op, "No need to reorder.");
  }

  // Axes are out of order, swap them.
  Value value = SqueezeTensorValue(rewriter, inner.getSrc(), op.getAxis());
  rewriter.modifyOpInPlace(op, [&]() {
    op.setOperand(value);
    op.setAxis(inner.getAxis() - 1);
  });
  return success();
}

// Converts squeeze_dims to tt.reshape.
LogicalResult SqueezeDimsToReshape(SqueezeDimsOp op,
                                   PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getSrc());
  return success();
}

// This pass removes unit dimensions from tensors to work around limitations of
// Triton's layout propagation and generate faster code.
//
// - squeeze_dims ops are extracted from tt.store and tt.reshape with unit
//   dimensions.
// - A series of patterns push squeeze_dims up through the graph, past
//   pure ops like broadcast, transpose, join, reduce.
// - Patterns for element-wise and load, which are non-pure ops that should
// not
//   be duplicated. This requires that all users are equivalent squeeze_dims.
// - squeeze_dims are folded into load and expand_dims, or converted back to
//   reshape.
class TritonXLASqueezeDimsPass
    : public impl::TritonXLASqueezeDimsPassBase<TritonXLASqueezeDimsPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(FoldSqueezeDimsOfLoad);
    patterns.add(SqueezeStore);
    patterns.add(SqueezeReshapeOperand);
    patterns.add(ExpandReshapeResult);
    patterns.add<PushSqueezeDimsUpThroughElementwise>(&getContext());
    patterns.add(PushSqueezeDimsUpThroughBroadcast);
    patterns.add(PushSqueezeDimsUpThroughTrans);
    patterns.add(PushSqueezeDimsUpThroughJoin);
    patterns.add(PushSqueezeDimsUpThroughReduce);
    patterns.add(PushSqueezeDimsUpThroughExpandDims);
    patterns.add(ReorderSqueezeDims);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    if (finalize_) {
      RewritePatternSet patterns(&getContext());
      patterns.add(SqueezeDimsToReshape);
      walkAndApplyPatterns(getOperation(), std::move(patterns));
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLASqueezeDimsPass() {
  return std::make_unique<TritonXLASqueezeDimsPass>();
}

}  // namespace mlir::triton::xla
