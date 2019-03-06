//===- Helpers.cpp - MLIR Declarative Helper Functionality ------*- C++ -*-===//
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
#include "mlir/IR/AffineExpr.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;
using namespace mlir::edsc;

static SmallVector<IndexHandle, 8> getMemRefSizes(Value *memRef) {
  MemRefType memRefType = memRef->getType().cast<MemRefType>();

  auto maps = memRefType.getAffineMaps();
  assert((maps.empty() || (maps.size() == 1 && maps[0].isIdentity())) &&
         "Layout maps not supported");
  SmallVector<IndexHandle, 8> res;
  res.reserve(memRefType.getShape().size());
  const auto &shape = memRefType.getShape();
  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (shape[idx] == -1) {
      res.push_back(IndexHandle(ValueHandle::create<DimOp>(memRef, idx)));
    } else {
      res.push_back(IndexHandle(static_cast<index_t>(shape[idx])));
    }
  }
  return res;
}

mlir::edsc::MemRefView::MemRefView(Value *v) : base(v) {
  assert(v->getType().isa<MemRefType>() && "MemRefType expected");

  auto memrefSizeValues = getMemRefSizes(v);
  for (auto &size : memrefSizeValues) {
    lbs.push_back(IndexHandle(static_cast<index_t>(0)));
    ubs.push_back(size);
    steps.push_back(1);
  }
}

/// Operator overloadings.
ValueHandle mlir::edsc::IndexedValue::operator+(ValueHandle e) {
  using op::operator+;
  return static_cast<ValueHandle>(*this) + e;
}
ValueHandle mlir::edsc::IndexedValue::operator-(ValueHandle e) {
  using op::operator-;
  return static_cast<ValueHandle>(*this) - e;
}
ValueHandle mlir::edsc::IndexedValue::operator*(ValueHandle e) {
  using op::operator*;
  return static_cast<ValueHandle>(*this) * e;
}
ValueHandle mlir::edsc::IndexedValue::operator/(ValueHandle e) {
  using op::operator/;
  return static_cast<ValueHandle>(*this) / e;
}

ValueHandle mlir::edsc::IndexedValue::operator+=(ValueHandle e) {
  using op::operator+;
  return intrinsics::STORE(*this + e, getBase(), indices);
}
ValueHandle mlir::edsc::IndexedValue::operator-=(ValueHandle e) {
  using op::operator-;
  return intrinsics::STORE(*this - e, getBase(), indices);
}
ValueHandle mlir::edsc::IndexedValue::operator*=(ValueHandle e) {
  using op::operator*;
  return intrinsics::STORE(*this * e, getBase(), indices);
}
ValueHandle mlir::edsc::IndexedValue::operator/=(ValueHandle e) {
  using op::operator/;
  return intrinsics::STORE(*this / e, getBase(), indices);
}

ValueHandle mlir::edsc::operator+(ValueHandle v, IndexedValue i) {
  using op::operator+;
  return v + static_cast<ValueHandle>(i);
}

ValueHandle mlir::edsc::operator-(ValueHandle v, IndexedValue i) {
  using op::operator-;
  return v - static_cast<ValueHandle>(i);
}

ValueHandle mlir::edsc::operator*(ValueHandle v, IndexedValue i) {
  using op::operator*;
  return v * static_cast<ValueHandle>(i);
}

ValueHandle mlir::edsc::operator/(ValueHandle v, IndexedValue i) {
  using op::operator/;
  return v / static_cast<ValueHandle>(i);
}
