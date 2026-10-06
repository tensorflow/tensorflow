/* Copyright 2026 The OpenXLA Authors.

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

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DEF_LEGALIZENARROWFLOATSTORAGEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace shlo = ::mlir::stablehlo;
namespace tensor = ::mlir::tensor;

using ::mlir::IntegerType;
using ::mlir::Location;
using ::mlir::LogicalResult;
using ::mlir::MemRefType;
using ::mlir::OpBuilder;
using ::mlir::PatternRewriter;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::Value;

// Returns the integer type used to carry values of `type` through data
// movement, or nullptr if `type` is not a narrow float.
IntegerType GetStorageType(Type type) {
  auto float_type = mlir::dyn_cast<mlir::FloatType>(type);
  if (float_type == nullptr) {
    return {};
  }
  unsigned width = float_type.getWidth();
  if (width != 8 && width != 16) {
    return {};
  }
  return IntegerType::get(type.getContext(), width);
}

// Same shape as `type`, with the element type replaced by its storage type, or
// nullptr if the element type is not a narrow float.
RankedTensorType GetStorageTensorType(Type type) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(type);
  if (tensor_type == nullptr) {
    return {};
  }
  IntegerType storage = GetStorageType(tensor_type.getElementType());
  if (storage == nullptr) {
    return {};
  }
  return tensor_type.clone(storage);
}

MemRefType GetStorageMemRefType(MemRefType type) {
  IntegerType storage = GetStorageType(type.getElementType());
  if (!storage) {
    return nullptr;
  }
  return MemRefType::Builder(type).setElementType(storage);
}

// Bitcasts a tensor or scalar `value` to `type`.
Value Bitcast(OpBuilder& builder, Location loc, Type type, Value value) {
  if (value.getType() == type) {
    return value;
  }
  if (mlir::isa<RankedTensorType>(type)) {
    return tensor::BitcastOp::create(builder, loc, type, value);
  }
  return ma::BitcastOp::create(builder, loc, type, value);
}

// Maps `value` from its narrow float type to the storage type.
Value ToStorage(OpBuilder& builder, Location loc, Value value) {
  Type type = value.getType();
  if (RankedTensorType tensor_type = GetStorageTensorType(type)) {
    return Bitcast(builder, loc, tensor_type, value);
  }
  if (IntegerType scalar_type = GetStorageType(type)) {
    return Bitcast(builder, loc, scalar_type, value);
  }
  return value;
}

// op(x : fN) -> bitcast(op(bitcast(x) : iN) : iN -> fN)
template <typename OpTy>
LogicalResult RetypeDataMovement(OpTy op, PatternRewriter& rewriter) {
  Type type = op.getType();
  RankedTensorType storage_type = GetStorageTensorType(type);
  if (!storage_type) {
    return rewriter.notifyMatchFailure(op, "not a narrow float tensor.");
  }
  llvm::SmallVector<Value> operands = llvm::map_to_vector(
      op->getOperands(),
      [&](Value operand) { return ToStorage(rewriter, op.getLoc(), operand); });
  auto storage_op = OpTy::create(rewriter, op.getLoc(), storage_type, operands,
                                 op.getProperties(),
                                 op->getDiscardableAttrDictionary().getValue());
  rewriter.replaceOpWithNewOp<tensor::BitcastOp>(op, type, storage_op);
  return mlir::success();
}

// bitcast(bitcast(x)) -> x, if the types round-trip.
LogicalResult FoldBitcastRoundTrip(tensor::BitcastOp bitcast,
                                   PatternRewriter& rewriter) {
  auto producer = bitcast.getSource().getDefiningOp<tensor::BitcastOp>();
  if (!producer || producer.getSource().getType() != bitcast.getType()) {
    return rewriter.notifyMatchFailure(bitcast, "not a round trip.");
  }
  rewriter.replaceOp(bitcast, producer.getSource());
  return mlir::success();
}

// Returns a view of narrow float `buffer` as its storage type. The view is
// created right after the definition of `buffer`, so it dominates every use of
// `buffer` and can be shared by all its tile accesses.
Value GetStorageBuffer(PatternRewriter& rewriter,
                       mlir::TypedValue<MemRefType> buffer) {
  MemRefType storage_type = GetStorageMemRefType(buffer.getType());
  for (mlir::Operation* user : buffer.getUsers()) {
    auto view = mlir::dyn_cast<xtile::MemRefBitcastOp>(user);
    if (view && view.getType() == storage_type) {
      return view;
    }
  }
  OpBuilder::InsertionGuard guard(rewriter);
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(buffer)) {
    rewriter.setInsertionPointToStart(arg.getOwner());
  } else {
    rewriter.setInsertionPointAfterValue(buffer);
  }
  return xtile::MemRefBitcastOp::create(rewriter, buffer.getLoc(), storage_type,
                                        buffer);
}

// extract(buffer : fN) -> bitcast(extract(view(buffer) : iN) : iN -> fN)
LogicalResult RetypeExtractTile(xtile::ExtractTileOp op,
                                PatternRewriter& rewriter) {
  if (!GetStorageMemRefType(op.getSource().getType())) {
    return rewriter.notifyMatchFailure(op, "not a narrow float buffer.");
  }
  Value buffer = GetStorageBuffer(rewriter, op.getSource());
  auto storage_tile = xtile::ExtractTileOp::create(
      rewriter, op.getLoc(), GetStorageTensorType(op.getType()), buffer,
      op.getOffsets(), op.getFullTileShapeAttr(), op.getStridesAttr());
  storage_tile->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  rewriter.replaceOpWithNewOp<tensor::BitcastOp>(op, op.getType(),
                                                 storage_tile);
  return mlir::success();
}

// insert(tile : fN, buffer : fN) -> insert(bitcast(tile) : iN, view(buffer))
LogicalResult RetypeInsertTile(xtile::InsertTileOp op,
                               PatternRewriter& rewriter) {
  if (!GetStorageMemRefType(op.getDestination().getType())) {
    return rewriter.notifyMatchFailure(op, "not a narrow float buffer.");
  }
  Value buffer = GetStorageBuffer(rewriter, op.getDestination());
  Value storage_tile = ToStorage(rewriter, op.getLoc(), op.getSource());
  rewriter.modifyOpInPlace(op, [&] {
    op.getSourceMutable().assign(storage_tile);
    op.getDestinationMutable().assign(buffer);
  });
  return mlir::success();
}

class LegalizeNarrowFloatStoragePass
    : public impl::LegalizeNarrowFloatStoragePassBase<
          LegalizeNarrowFloatStoragePass> {
 public:
  using LegalizeNarrowFloatStoragePassBase::LegalizeNarrowFloatStoragePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add(RetypeDataMovement<shlo::TransposeOp>);
    patterns.add(RetypeDataMovement<shlo::ReshapeOp>);
    patterns.add(RetypeDataMovement<shlo::BroadcastInDimOp>);
    patterns.add(RetypeDataMovement<shlo::SliceOp>);
    patterns.add(RetypeDataMovement<shlo::ConcatenateOp>);
    patterns.add(RetypeDataMovement<xtile::MaskOp>);
    patterns.add(RetypeDataMovement<ma::SelectOp>);
    patterns.add(FoldBitcastRoundTrip);
    patterns.add(RetypeExtractTile);
    patterns.add(RetypeInsertTile);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace xla::cpu
