//===- Intrinsics.cpp - MLIR Operations for Declarative Builders *- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"

using namespace mlir;
using namespace mlir::edsc;

InstructionHandle mlir::edsc::intrinsics::BR(BlockHandle bh,
                                             ArrayRef<ValueHandle> operands) {
  assert(bh && "Expected already captured BlockHandle");
  for (auto &o : operands) {
    (void)o;
    assert(o && "Expected already captured ValueHandle");
  }
  SmallVector<Value *, 4> ops(operands.begin(), operands.end());
  return InstructionHandle::create<BranchOp>(bh.getBlock(), ops);
}
static void enforceEmptyCapturesMatchOperands(ArrayRef<ValueHandle *> captures,
                                              ArrayRef<ValueHandle> operands) {
  assert(captures.size() == operands.size() &&
         "Expected same number of captures as operands");
  for (auto it : llvm::zip(captures, operands)) {
    (void)it;
    assert(!std::get<0>(it)->hasValue() &&
           "Unexpected already captured ValueHandle");
    assert(std::get<1>(it) && "Expected already captured ValueHandle");
    assert(std::get<0>(it)->getType() == std::get<1>(it).getType() &&
           "Expected the same type for capture and operand");
  }
}

InstructionHandle mlir::edsc::intrinsics::BR(BlockHandle *bh,
                                             ArrayRef<ValueHandle *> captures,
                                             ArrayRef<ValueHandle> operands) {
  assert(!*bh && "Unexpected already captured BlockHandle");
  enforceEmptyCapturesMatchOperands(captures, operands);
  BlockBuilder(bh, captures)({/* no body */});
  SmallVector<Value *, 4> ops(operands.begin(), operands.end());
  return InstructionHandle::create<BranchOp>(bh->getBlock(), ops);
}

InstructionHandle
mlir::edsc::intrinsics::COND_BR(ValueHandle cond, BlockHandle trueBranch,
                                ArrayRef<ValueHandle> trueOperands,
                                BlockHandle falseBranch,
                                ArrayRef<ValueHandle> falseOperands) {
  SmallVector<Value *, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value *, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return InstructionHandle::create<CondBranchOp>(
      cond, trueBranch.getBlock(), trueOps, falseBranch.getBlock(), falseOps);
}

InstructionHandle mlir::edsc::intrinsics::COND_BR(
    ValueHandle cond, BlockHandle *trueBranch,
    ArrayRef<ValueHandle *> trueCaptures, ArrayRef<ValueHandle> trueOperands,
    BlockHandle *falseBranch, ArrayRef<ValueHandle *> falseCaptures,
    ArrayRef<ValueHandle> falseOperands) {
  assert(!*trueBranch && "Unexpected already captured BlockHandle");
  assert(!*falseBranch && "Unexpected already captured BlockHandle");
  enforceEmptyCapturesMatchOperands(trueCaptures, trueOperands);
  enforceEmptyCapturesMatchOperands(falseCaptures, falseOperands);
  BlockBuilder(trueBranch, trueCaptures)({/* no body */});
  BlockBuilder(falseBranch, falseCaptures)({/* no body */});
  SmallVector<Value *, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value *, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return InstructionHandle::create<CondBranchOp>(
      cond, trueBranch->getBlock(), trueOps, falseBranch->getBlock(), falseOps);
}

////////////////////////////////////////////////////////////////////////////////
// TODO(ntv): Intrinsics below this line should be TableGen'd.
////////////////////////////////////////////////////////////////////////////////
ValueHandle
mlir::edsc::intrinsics::LOAD(ValueHandle base,
                             llvm::ArrayRef<ValueHandle> indices = {}) {
  SmallVector<Value *, 4> ops(indices.begin(), indices.end());
  return ValueHandle::create<LoadOp>(base.getValue(), ops);
}

InstructionHandle
mlir::edsc::intrinsics::RETURN(ArrayRef<ValueHandle> operands) {
  SmallVector<Value *, 4> ops(operands.begin(), operands.end());
  return InstructionHandle::create<ReturnOp>(ops);
}

InstructionHandle
mlir::edsc::intrinsics::STORE(ValueHandle value, ValueHandle base,
                              llvm::ArrayRef<ValueHandle> indices = {}) {
  SmallVector<Value *, 4> ops(indices.begin(), indices.end());
  return InstructionHandle::create<StoreOp>(value.getValue(), base.getValue(),
                                            ops);
}
