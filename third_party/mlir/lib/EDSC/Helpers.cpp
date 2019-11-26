//===- Helpers.cpp - MLIR Declarative Helper Functionality ----------------===//
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

#include "mlir/EDSC/Helpers.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::edsc;

static SmallVector<ValueHandle, 8> getMemRefSizes(Value *memRef) {
  MemRefType memRefType = memRef->getType().cast<MemRefType>();
  assert(isStrided(memRefType) && "Expected strided MemRef type");

  SmallVector<ValueHandle, 8> res;
  res.reserve(memRefType.getShape().size());
  const auto &shape = memRefType.getShape();
  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (shape[idx] == -1) {
      res.push_back(ValueHandle::create<DimOp>(memRef, idx));
    } else {
      res.push_back(static_cast<index_t>(shape[idx]));
    }
  }
  return res;
}

mlir::edsc::MemRefView::MemRefView(Value *v) : base(v) {
  assert(v->getType().isa<MemRefType>() && "MemRefType expected");

  auto memrefSizeValues = getMemRefSizes(v);
  for (auto &size : memrefSizeValues) {
    lbs.push_back(static_cast<index_t>(0));
    ubs.push_back(size);
    steps.push_back(1);
  }
}

mlir::edsc::VectorView::VectorView(Value *v) : base(v) {
  auto vectorType = v->getType().cast<VectorType>();

  for (auto s : vectorType.getShape()) {
    lbs.push_back(static_cast<index_t>(0));
    ubs.push_back(static_cast<index_t>(s));
    steps.push_back(1);
  }
}
