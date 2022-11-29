/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file provides basic utilities for the elemental lowering of
// each node

#include "lhlo/transforms/lhlo_elemental_utils.h"

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "utils/codegen_utils.h"

using mlir::memref::DimOp;
using mlir::memref::LoadOp;
using mlir::memref::StoreOp;

namespace mlir {
namespace lmhlo {

Value createLoadOrUseCachedValue(Location loc, OpBuilder* b, Value memref,
                                 ValueRange indices,
                                 OpBuilder::InsertPoint insertPoint) {
  // Check if there are any cached value that can be reused,
  // within the current Block. Alternatively we can do this for
  // all the Blocks that dominant this Block, but that will be
  // complicated anyway.
  std::vector<StoreOp> storeOps;
  insertPoint.getBlock()->walk(
      insertPoint.getBlock()->begin(), insertPoint.getPoint(),
      [&](StoreOp storeOp) {
        if (storeOp.getOperation()->getBlock() != insertPoint.getBlock())
          return;
        if ((storeOp.getMemRef() == memref) &&
            (storeOp.getIndices() == indices))
          storeOps.emplace_back(storeOp);
      });
  if (!storeOps.empty()) return storeOps[0].getOperand(0);
  int rank = memref.getType().dyn_cast<MemRefType>().getRank();
  return rank > 0 ? b->create<LoadOp>(loc, memref, indices)
                  : b->create<LoadOp>(loc, memref);
}

DenseSet<Operation*> noLoaderUser(SmallVectorImpl<Operation*>& ops) {
  SmallVector<Operation*, 4> worklist;
  DenseSet<Operation*> hasLoaderOps;
  for (Operation* op : ops) {
    Value memref = cast<LmhloOp>(op).getResultBuffer();
    if (memref == nullptr) continue;
    for (auto* user : memref.getUsers()) {
      if (isa<memref::LoadOp>(user)) {
        worklist.push_back(op);
        hasLoaderOps.insert(op);
      }
    }
  }

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    int numOperands = op->getNumOperands();
    for (int i = 0; i < numOperands - 1; ++i) {
      Value memref = op->getOperand(i);
      for (Operation* user : memref.getUsers()) {
        if ((!isa<LmhloOp>(user)) || hasLoaderOps.count(user)) continue;
        if (cast<LmhloOp>(user).getResultBuffer() == memref) {
          worklist.push_back(user);
          hasLoaderOps.insert(user);
        }
      }
    }
  }

  DenseSet<Operation*> noLoaderOps;
  for (Operation* op : ops)
    if (!hasLoaderOps.count(op)) noLoaderOps.insert(op);
  return noLoaderOps;
}

void cleanUnusedLhloOps(Block* parent) {
  SmallVector<Operation*, 4> lhloOps;
  for (Operation& op : parent->getOperations()) {
    if (op.getDialect() == op.getContext()->getLoadedDialect("lmhlo") &&
        (!isa<lmhlo::TerminatorOp>(op)))
      lhloOps.push_back(&op);
  }
  for (auto* lhloOp : noLoaderUser(lhloOps)) lhloOp->erase();
}

template <typename LHLO_OpTy>
Value elementalLower(OpBuilder* b, Location loc, LHLO_OpTy op,
                     ValueRange outputIndex, bool checkCache);

template <>
Value elementalLower<lmhlo::RealDynamicSliceOp>(OpBuilder* b, Location loc,
                                                lmhlo::RealDynamicSliceOp op,
                                                ValueRange outputIndex,
                                                bool checkCache) {
  Value startIndicesMemref = op->getOperand(1);
  Value stridesMemref = op->getOperand(3);
  int rank = outputIndex.size();
  SmallVector<Value, 4> inputIndex;
  for (int dim = 0; dim < rank; ++dim) {
    SmallVector<Value, 4> dimIndex;
    dimIndex.push_back(b->create<arith::ConstantOp>(
        loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), dim)));
    auto startIndexLoad =
        b->create<LoadOp>(loc, startIndicesMemref, ValueRange{dimIndex});
    auto startIndex =
        b->create<arith::IndexCastOp>(loc, b->getIndexType(), startIndexLoad);
    auto strideLoad =
        b->create<LoadOp>(loc, stridesMemref, ValueRange{dimIndex});
    auto stride =
        b->create<arith::IndexCastOp>(loc, b->getIndexType(), strideLoad);
    // input_dim = out_dim * stride + start_index
    auto inputDim = b->create<arith::AddIOp>(
        loc, b->create<arith::MulIOp>(loc, outputIndex[dim], stride),
        startIndex);
    inputIndex.push_back(inputDim);
  }

  Value operandMemref = *(op->getOperands().begin());

  if (!checkCache) return b->create<LoadOp>(loc, operandMemref, inputIndex);
  return createLoadOrUseCachedValue(loc, b, operandMemref, inputIndex,
                                    b->saveInsertionPoint());
}

namespace {

template <typename T>
Value elementalLowerImplForBroadcastInDimOps(OpBuilder* b, Location loc,
                                             T broadcastInDim,
                                             ValueRange outputIndex,
                                             bool checkCache) {
  auto broadcastDimensions =
      broadcastInDim.getBroadcastDimensions().template getValues<int64_t>();
  int outRank = outputIndex.size();
  Value operandMemref = broadcastInDim->getOperand(0);
  SmallVector<Value, 4> inputIndex;
  for (int64_t dim = 0; dim < outRank; ++dim) {
    auto it =
        std::find(broadcastDimensions.begin(), broadcastDimensions.end(), dim);

    bool isBroadcastDim = (it != broadcastDimensions.end());
    if (isBroadcastDim) {
      int inputDim = std::distance(broadcastDimensions.begin(), it);
      int64_t staticDimSize =
          operandMemref.getType().cast<MemRefType>().getShape()[inputDim];
      if (staticDimSize == 1) {
        // we know this dim is to be broadcasted at compile time
        auto zero = b->create<arith::ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        inputIndex.push_back(zero);
      } else if (staticDimSize == ShapedType::kDynamic) {
        // we are not sure if this dim is to be broadcasted at compile time
        auto dimSize = b->create<DimOp>(loc, operandMemref, inputDim);
        auto one = b->create<arith::ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 1));
        auto zero = b->create<arith::ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        auto dimSizeIs1 = b->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, dimSize, one);
        inputIndex.push_back(b->create<mlir::arith::SelectOp>(
            loc, dimSizeIs1, zero, outputIndex[dim]));
      } else {
        // we know this dim is not to be broadcasted at compile time
        inputIndex.push_back(outputIndex[dim]);
      }
    }
  }

  if (!checkCache) {
    int rank = operandMemref.getType().dyn_cast<MemRefType>().getRank();
    return (rank > 0) ? b->create<LoadOp>(loc, operandMemref, inputIndex)
                      : b->create<LoadOp>(loc, operandMemref, ValueRange());
  }
  return createLoadOrUseCachedValue(loc, b, operandMemref, inputIndex,
                                    b->saveInsertionPoint());
}

}  // namespace

template <>
Value elementalLower<lmhlo::DynamicBroadcastInDimOp>(
    OpBuilder* b, Location loc, lmhlo::DynamicBroadcastInDimOp op,
    ValueRange outputIndex, bool checkCache) {
  return elementalLowerImplForBroadcastInDimOps(b, loc, op, outputIndex,
                                                checkCache);
}

template <>
Value elementalLower<lmhlo::BroadcastInDimOp>(OpBuilder* b, Location loc,
                                              lmhlo::BroadcastInDimOp op,
                                              ValueRange outputIndex,
                                              bool checkCache) {
  return elementalLowerImplForBroadcastInDimOps(b, loc, op, outputIndex,
                                                checkCache);
}

scf::ForOp createLoopAndSetInsPt(OpBuilder& b, Location loc, Value& var,
                                 Value lb, Value ub, Value step,
                                 ArrayRef<Value> initValues) {
  auto forOp = b.create<scf::ForOp>(loc, lb, ub, step, initValues);
  b.setInsertionPointToStart(forOp.getBody());
  var = forOp.getInductionVar();
  return forOp;
}

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> initValues) {
  auto parOp = b.create<scf::ParallelOp>(loc, lbs, ubs, steps, initValues,
                                         /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(parOp.getBody());
  vars.append(parOp.getInductionVars().begin(), parOp.getInductionVars().end());
  return parOp;
}

// reinterpret_cast the input memref into 1D
memref::ReinterpretCastOp createMemRef1DReinterpretCast(OpBuilder& b,
                                                        Location loc,
                                                        Value memref) {
  auto memrefTy = memref.getType().cast<MemRefType>();
  assert(memrefTy.getLayout().isIdentity());
  Value size = codegen_utils::emitNumElementsComputation(b, loc, memref);
  Value stride = b.create<mlir::arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  Value zero = b.create<mlir::arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));
  auto memref1dType =
      MemRefType::get({ShapedType::kDynamic}, memrefTy.getElementType(),
                      b.getMultiDimIdentityMap(1), memrefTy.getMemorySpace());
  return b.create<memref::ReinterpretCastOp>(
      loc, memref1dType, memref, zero, ValueRange{size}, ValueRange{stride});
}

void createOffsetStore(OpBuilder& b, Location loc, Value res, Value memref,
                       Value offset) {
  Value memref1d = createMemRef1DReinterpretCast(b, loc, memref);
  b.create<memref::StoreOp>(loc, res, memref1d, ValueRange{offset});
}

memref::LoadOp createOffsetLoad(OpBuilder& b, Location loc, Value memref,
                                Value offset) {
  Value memref1d = createMemRef1DReinterpretCast(b, loc, memref);
  return b.create<memref::LoadOp>(loc, memref1d, ValueRange{offset});
}

}  // namespace lmhlo
}  // namespace mlir
