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

#include "gml_st/transforms/transforms.h"

#include <cstddef>
#include <tuple>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {

bool isZero(Value v) { return matchPattern(v, m_Zero()); }
bool isOne(Value v) { return matchPattern(v, m_One()); }

bool hasSingleElementOperandsAndResults(Operation *op) {
  auto isScalar = [](Type type) {
    return !type.isa<mlir::ShapedType>() ||
           (type.isa<TensorType>() &&
            hasSingleElement(type.cast<TensorType>()));
  };
  return llvm::all_of(op->getOperandTypes(), isScalar) &&
         llvm::all_of(op->getResultTypes(), isScalar);
}

/// Hoisting after vectorization
namespace {

using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

bool isLoopInvariantTransferWriteOp(ForOp forOp, TransferWriteOp candidate) {
  // Indexing must not depend on `forOp`.
  for (Value operand : candidate.getIndices())
    if (!forOp.isDefinedOutsideOfLoop(operand)) return false;
  return candidate->hasOneUse();
}

/// Look for a TransferReadOp, in the given tensor users, accessing the same
/// offset as `write`.
FailureOr<TransferReadOp> findMatchingTransferRead(TransferWriteOp write,
                                                   Value srcTensor) {
  SmallVector<Operation *> users(srcTensor.getUsers().begin(),
                                 srcTensor.getUsers().end());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();

    auto read = dyn_cast<vector::TransferReadOp>(user);
    if (read && read.getIndices() == write.getIndices() &&
        read.getVectorType() == write.getVectorType())
      return read;
  }
  return failure();
}

/// Check if the chunk of data inserted by `write` is read by any
/// other op than `candidateRead` or `terminator`.
bool tensorChunkAccessedByUnknownOp(TransferWriteOp write,
                                    TransferReadOp candidateRead, Value tensor,
                                    SetYieldOp terminator) {
  // Make sure none of the other uses read the part of the tensor modified
  // by the transfer_write.
  llvm::SmallVector<Value::use_range, 1> uses;
  uses.push_back(tensor.getUses());
  while (!uses.empty()) {
    for (OpOperand &use : uses.pop_back_val()) {
      Operation *user = use.getOwner();
      // Skip the candidate and terminator uses, only inspect the "other" uses.
      if (user == candidateRead || user == write || user == terminator)
        continue;
      // Consider all transitive uses through a extract_slice / insert_slice.
      // Consider all transitive uses through a vector.transfer_write.
      // Consider all nested uses through a gml_st::ForOp. We may have
      // pass-through tensor arguments left from previous level of hoisting.
      // TODO(vuson): atm we just bail because a stronger analysis is needed for
      // these cases.
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
              vector::TransferWriteOp, ForOp, SetYieldOp>(user))
        return true;

      auto read = dyn_cast<TransferReadOp>(user);
      if (!read || !vector::isDisjointTransferIndices(
                       cast<VectorTransferOpInterface>(read.getOperation()),
                       cast<VectorTransferOpInterface>(write.getOperation()))) {
        return true;
      }
    }
  }
  return false;
}

ForOp replaceLoopWithNewYields(OpBuilder &builder, ForOp loop,
                               ValueRange newOutputOperands,
                               ValueRange newYieldValues, Value yieldSet) {
  assert(newOutputOperands.size() == newYieldValues.size() &&
         "expected as many new yield values as new iter operands");
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loop);
  auto operands = llvm::to_vector(loop.getOutputs());
  operands.append(newOutputOperands.begin(), newOutputOperands.end());
  auto newLoop = builder.create<ForOp>(
      loop.getLoc(),
      llvm::to_vector<1>(llvm::map_range(
          operands, [&](Value v) -> Type { return v.getType(); })),
      loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), operands,
      nullptr);

  Block *loopBody = loop.getBody();
  Block *newLoopBody = newLoop.getBody();

  // Move the body of the original loop to the new loop.
  builder.setInsertionPointToStart(newLoopBody);
  IRMapping bvm;
  for (Operation &bodyMember : loopBody->without_terminator()) {
    builder.clone(bodyMember, bvm);
  }

  // Generate the new yield values to use by using the callback and append the
  // yield values to the set_yield operation.
  auto oldYield = loop.getTerminator();
  ArrayRef<BlockArgument> newBBArgs =
      newLoopBody->getArguments().take_back(newOutputOperands.size());
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(newLoopBody);
    auto getMappedValues = [&](ValueRange values) {
      return llvm::to_vector(llvm::map_range(
          values, [&](Value value) { return bvm.lookupOrDefault(value); }));
    };
    auto srcs = getMappedValues(oldYield.getSrcs());
    srcs.append(getMappedValues(newYieldValues));
    auto dsts = getMappedValues(oldYield.getDsts());
    dsts.append(newBBArgs.begin(), newBBArgs.end());
    auto sets = getMappedValues(oldYield.getSets());
    sets.append(newYieldValues.size(), bvm.lookupOrDefault(yieldSet));
    builder.create<SetYieldOp>(newLoop.getLoc(), srcs, dsts, sets);
  }

  // Remap the BlockArguments from the original loop to the new loop
  // BlockArguments.
  ArrayRef<BlockArgument> bbArgs = loopBody->getArguments();
  for (auto it :
       llvm::zip(bbArgs, newLoopBody->getArguments().take_front(bbArgs.size())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  // Replace all uses of `newOutputOperands` with the corresponding basic block
  // arguments.
  for (auto it : llvm::zip(newOutputOperands, newBBArgs)) {
    std::get<0>(it).replaceUsesWithIf(std::get<1>(it), [&](OpOperand &use) {
      Operation *user = use.getOwner();
      return newLoop->isProperAncestor(user);
    });
  }

  // Replace all uses of the original loop with corresponding values from the
  // new loop.
  loop.replaceAllUsesWith(
      newLoop.getResults().take_front(loop.getNumResults()));

  return newLoop;
}

/// Mechanical hoisting of a matching transfeSeread / transfer_write pair.
void hoistReadWrite(TransferReadOp read, TransferWriteOp write,
                    BlockArgument tensorBBArg, Value yieldSet) {
  auto forOp = cast<ForOp>(tensorBBArg.getOwner()->getParentOp());

  // Hoist the transfer_read op.
  forOp.moveOutOfLoop(read);

  // FIXME: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  unsigned initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // Update the source tensor.
  read.getSourceMutable().assign(forOp.getOutputs()[initArgNumber]);

  // Hoist write after.
  write->moveAfter(forOp);

  // Update the yield.
  auto setYieldOp = forOp.getTerminator();
  setYieldOp->setOperand(initArgNumber, write.getSource());

  // Rewrite `loop` with additional new yields.
  OpBuilder b(read);
  auto newForOp = replaceLoopWithNewYields(b, forOp, read.getVector(),
                                           write.getVector(), yieldSet);

  // Transfer write has been hoisted, need to update the vector and tensor
  // source. Replace the result of the loop to use the new tensor created
  // outside the loop.
  // Depending on whether a insert_slice is present or not, it carries the
  // update on the tensor operands.
  newForOp.getResult(initArgNumber).replaceAllUsesWith(write.getResult());
  write.getSourceMutable().assign(newForOp.getResult(initArgNumber));

  // Always update with the newly yield tensor and vector.
  write.getVectorMutable().assign(newForOp.getResults().back());
}
}  // namespace

bool isIdentitySlice(ValueRange offsets, ValueRange strides) {
  // Offsets must be all 0s and strides must be all 1s.
  return llvm::all_of(offsets, [](Value v) { return isZero(v); }) &&
         llvm::all_of(strides, [](Value v) { return isOne(v); });
}

bool haveSameStaticShape(Value lhs, Value rhs) {
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return false;
  return lhsType == rhsType;
}

void hoistRedundantVectorTransfersOnTensor(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](ForOp forOp) {
      auto terminator = forOp.getTerminator();
      for (const auto &[src, set, dst, outputArg] :
           llvm::zip(terminator.getSrcs(), terminator.getSets(),
                     terminator.getDsts(), forOp.getRegionOutputArgs())) {
        auto write = src.getDefiningOp<TransferWriteOp>();
        if (!write) continue;
        if (!isLoopInvariantTransferWriteOp(forOp, write)) continue;

        auto srcTensor = write.getSource();
        if (srcTensor != outputArg) continue;

        auto tileOp = set.getDefiningOp<TileOp>();
        if (!tileOp ||
            !isIdentitySlice(tileOp.getOffsets(), tileOp.getStrides()) ||
            !haveSameStaticShape(src, dst))
          continue;

        // Find a read with the same type and indices.
        auto matchingRead = findMatchingTransferRead(write, srcTensor);

        // Make sure none of the other uses reads the part of the tensor
        // modified by the transfer_write.
        if (failed(matchingRead) ||
            tensorChunkAccessedByUnknownOp(write, *matchingRead, srcTensor,
                                           terminator))
          continue;

        hoistReadWrite(*matchingRead, write, outputArg, set);
        changed = true;
        forOp.erase();

        // Need to interrupt and restart: erasing the loop messes up the walk.
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    // Apply canonicalization so the newForOp + yield folds immediately, thus
    // cleaning up the IR and potentially enabling more hoisting.
    if (changed) {
      auto *ctx = func->getContext();
      RewritePatternSet patterns(ctx);
      ForOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
  }
}

void setLabel(Operation *op, StringRef name) {
  op->setAttr(name, UnitAttr::get(op->getContext()));
}

void removeLabel(Operation *op, StringRef name) { op->removeAttr(name); }

bool hasLabel(Operation *op, StringRef name) { return op->hasAttr(name); }

constexpr llvm::StringLiteral kOpLabel = "op_label";

bool hasMatchingLabel(Operation *op, StringRef label) {
  auto opLabelAttr = op->getAttr(kOpLabel);
  if (!opLabelAttr) return false;

  return opLabelAttr.cast<StringAttr>().getValue() == label;
}

}  // namespace gml_st
}  // namespace mlir
