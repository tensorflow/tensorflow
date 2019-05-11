//===- Utils.cpp - Implementation of utiliy functions for Linalg ----------===//
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
// This file implements utility functions for the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Utils.h"
#include "linalg1/Intrinsics.h"
#include "linalg1/Ops.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::intrinsics;

unsigned linalg::getViewRank(Value *view) {
  assert(view->getType().isa<ViewType>() && "expected a ViewType");
  if (auto viewOp = dyn_cast<ViewOp>(view->getDefiningOp()))
    return viewOp.getRank();
  return view->getDefiningOp()->cast<SliceOp>().getRank();
}

ViewOp linalg::emitAndReturnViewOpFromMemRef(Value *memRef) {
  // Syntactic sugar helper to extract and emit view-like information from an
  // mlir::MemRef without boilerplate.
  mlir::edsc::MemRefView v(memRef);
  SmallVector<Value *, 8> indices(v.rank());
  for (unsigned i = 0; i < v.rank(); ++i) {
    indices[i] = range(v.lb(i), v.ub(i), constant_index(v.step(i)));
  }
  return ScopedContext::getBuilder()->create<ViewOp>(
      ScopedContext::getLocation(), memRef, indices);
}
