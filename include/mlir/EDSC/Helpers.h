//===- Helpers.h - MLIR Declarative Helper Functionality --------*- C++ -*-===//
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
// Provides helper classes and syntactic sugar for declarative builders.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDSC_HELPERS_H_
#define MLIR_EDSC_HELPERS_H_

#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace edsc {

class IndexedValue;

/// An IndexHandle is a simple wrapper around a ValueHandle.
/// IndexHandles are ubiquitous enough to justify a new type to allow simple
/// declarations without boilerplate such as:
///
/// ```c++
///    IndexHandle i, j, k;
/// ```
struct IndexHandle : public ValueHandle {
  explicit IndexHandle()
      : ValueHandle(ScopedContext::getBuilder()->getIndexType()) {}
  explicit IndexHandle(index_t v) : ValueHandle(v) {}
  explicit IndexHandle(Value *v) : ValueHandle(v) {
    assert(v->getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
  }
  explicit IndexHandle(ValueHandle v) : ValueHandle(v) {}
};

/// A MemRefView represents the information required to step through a
/// MemRef. It has placeholders for non-contiguous tensors that fit within the
/// Fortran subarray model.
/// At the moment it can only capture a MemRef with an identity layout map.
// TODO(ntv): Support MemRefs with layoutMaps.
class MemRefView {
public:
  explicit MemRefView(Value *v);
  MemRefView(const MemRefView &) = default;
  MemRefView &operator=(const MemRefView &) = default;

  unsigned rank() const { return lbs.size(); }
  unsigned fastestVarying() const { return rank() - 1; }

  std::tuple<IndexHandle, IndexHandle, int64_t> range(unsigned idx) {
    return std::make_tuple(lbs[idx], ubs[idx], steps[idx]);
  }

private:
  friend IndexedValue;

  ValueHandle base;
  SmallVector<IndexHandle, 8> lbs;
  SmallVector<IndexHandle, 8> ubs;
  SmallVector<int64_t, 8> steps;
};

ValueHandle operator+(ValueHandle v, IndexedValue i);
ValueHandle operator-(ValueHandle v, IndexedValue i);
ValueHandle operator*(ValueHandle v, IndexedValue i);
ValueHandle operator/(ValueHandle v, IndexedValue i);

/// This helper class is an abstraction over memref, that purely for sugaring
/// purposes and allows writing compact expressions such as:
///
/// ```mlir
///    IndexedValue A(...), B(...), C(...);
///    For(ivs, zeros, shapeA, ones, {
///      C(ivs) = A(ivs) + B(ivs)
///    });
/// ```
///
/// Assigning to an IndexedValue emits an actual store operation, while using
/// converting an IndexedValue to a ValueHandle emits an actual load operation.
struct IndexedValue {
  explicit IndexedValue(MemRefView &v, llvm::ArrayRef<ValueHandle> indices = {})
      : view(v), indices(indices.begin(), indices.end()) {}

  IndexedValue(const IndexedValue &rhs) = default;
  IndexedValue &operator=(const IndexedValue &rhs) = default;

  /// Returns a new `IndexedValue`.
  IndexedValue operator()(llvm::ArrayRef<ValueHandle> indices = {}) {
    return IndexedValue(view, indices);
  }

  /// Emits a `store`.
  // NOLINTNEXTLINE: unconventional-assign-operator
  InstructionHandle operator=(ValueHandle rhs) {
    return intrinsics::STORE(rhs, getBase(), indices);
  }

  ValueHandle getBase() const { return view.base; }

  /// Emits a `load` when converting to a ValueHandle.
  explicit operator ValueHandle() {
    return intrinsics::LOAD(getBase(), indices);
  }

  /// Operator overloadings.
  ValueHandle operator+(ValueHandle e);
  ValueHandle operator-(ValueHandle e);
  ValueHandle operator*(ValueHandle e);
  ValueHandle operator/(ValueHandle e);
  InstructionHandle operator+=(ValueHandle e);
  InstructionHandle operator-=(ValueHandle e);
  InstructionHandle operator*=(ValueHandle e);
  InstructionHandle operator/=(ValueHandle e);
  ValueHandle operator+(IndexedValue e) {
    return *this + static_cast<ValueHandle>(e);
  }
  ValueHandle operator-(IndexedValue e) {
    return *this - static_cast<ValueHandle>(e);
  }
  ValueHandle operator*(IndexedValue e) {
    return *this * static_cast<ValueHandle>(e);
  }
  ValueHandle operator/(IndexedValue e) {
    return *this / static_cast<ValueHandle>(e);
  }
  InstructionHandle operator+=(IndexedValue e) {
    return this->operator+=(static_cast<ValueHandle>(e));
  }
  InstructionHandle operator-=(IndexedValue e) {
    return this->operator-=(static_cast<ValueHandle>(e));
  }
  InstructionHandle operator*=(IndexedValue e) {
    return this->operator*=(static_cast<ValueHandle>(e));
  }
  InstructionHandle operator/=(IndexedValue e) {
    return this->operator/=(static_cast<ValueHandle>(e));
  }

private:
  MemRefView &view;
  llvm::SmallVector<ValueHandle, 8> indices;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_HELPERS_H_
