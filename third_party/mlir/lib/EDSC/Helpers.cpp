//===- Helpers.cpp - MLIR Declarative Helper Functionality ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/EDSC/Helpers.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::edsc;

static SmallVector<ValueHandle, 8> getMemRefSizes(Value memRef) {
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

mlir::edsc::MemRefView::MemRefView(Value v) : base(v) {
  assert(v->getType().isa<MemRefType>() && "MemRefType expected");

  auto memrefSizeValues = getMemRefSizes(v);
  for (auto &size : memrefSizeValues) {
    lbs.push_back(static_cast<index_t>(0));
    ubs.push_back(size);
    steps.push_back(1);
  }
}

mlir::edsc::VectorView::VectorView(Value v) : base(v) {
  auto vectorType = v->getType().cast<VectorType>();

  for (auto s : vectorType.getShape()) {
    lbs.push_back(static_cast<index_t>(0));
    ubs.push_back(static_cast<index_t>(s));
    steps.push_back(1);
  }
}
