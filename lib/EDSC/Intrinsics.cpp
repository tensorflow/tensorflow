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

using namespace mlir;
using namespace mlir::edsc;

ValueHandle mlir::edsc::intrinsics::BR(BlockHandle bh,
                                       ArrayRef<ValueHandle> operands) {
  assert(bh && "Expected already captured BlockHandle");
  for (auto &o : operands) {
    assert(o && "Expected already captured ValueHandle");
  }
  SmallVector<Value *, 4> ops(operands.begin(), operands.end());
  return ValueHandle::create<BranchOp>(bh.getBlock(), ops);
}

ValueHandle
mlir::edsc::intrinsics::COND_BR(ValueHandle cond, BlockHandle trueBranch,
                                ArrayRef<ValueHandle> trueOperands,
                                BlockHandle falseBranch,
                                ArrayRef<ValueHandle> falseOperands) {
  SmallVector<Value *, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value *, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return ValueHandle::create<CondBranchOp>(cond, trueBranch.getBlock(), trueOps,
                                           falseBranch.getBlock(), falseOps);
}

////////////////////////////////////////////////////////////////////////////////
// TODO(ntv): Intrinsics below this line should be TableGen'd.
////////////////////////////////////////////////////////////////////////////////
ValueHandle mlir::edsc::intrinsics::RETURN(ArrayRef<ValueHandle> operands) {
  SmallVector<Value *, 4> ops(operands.begin(), operands.end());
  return ValueHandle::create<ReturnOp>(ops);
}
