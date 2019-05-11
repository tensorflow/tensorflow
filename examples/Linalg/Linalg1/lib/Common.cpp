//===- Common.cpp - Implementation of common supporting functions ---------===//
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
//
// This file implements a simple IR operation to create a new RangeType in the
// linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Common.h"
#include "linalg1/Ops.h"
#include "linalg1/Types.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/StandardOps/Ops.h"

using llvm::ArrayRef;
using mlir::ConstantIndexOp;
using mlir::edsc::CapturableHandle;
using mlir::edsc::ValueHandle;
using mlir::edsc::intrinsics::alloc;
using mlir::edsc::intrinsics::ret;

using namespace linalg;

linalg::common::LoopNestRangeBuilder::LoopNestRangeBuilder(
    llvm::ArrayRef<ValueHandle *> ivs, llvm::ArrayRef<ValueHandle> indexings) {
  assert(ivs.size() == indexings.size());
  for (unsigned i = 0, e = indexings.size(); i < e; ++i) {
    auto rangeOp =
        llvm::dyn_cast<RangeOp>(indexings[i].getValue()->getDefiningOp());
    if (!rangeOp) {
      continue;
    }
    auto lb = rangeOp.getMin();
    auto ub = rangeOp.getMax();
    // This must be a constexpr index until we relax the affine.for constraint
    auto step =
        rangeOp.getStep()->getDefiningOp()->cast<ConstantIndexOp>().getValue();
    loops.emplace_back(ivs[i], ValueHandle(lb), ValueHandle(ub), step);
  }
}

linalg::common::LoopNestRangeBuilder::LoopNestRangeBuilder(
    llvm::ArrayRef<ValueHandle *> ivs, llvm::ArrayRef<mlir::Value *> indexings)
    : LoopNestRangeBuilder(ivs, llvm::SmallVector<ValueHandle, 4>(
                                    indexings.begin(), indexings.end())) {}

ValueHandle linalg::common::LoopNestRangeBuilder::operator()(
    llvm::ArrayRef<CapturableHandle> stmts) {
  for (auto &lit : llvm::reverse(loops)) {
    lit({});
  }
  return ValueHandle::null();
}
