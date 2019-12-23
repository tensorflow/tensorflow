//===- Helpers.h - MLIR Declarative Helper Functionality --------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

// A TemplatedIndexedValue brings an index notation over the template Load and
// Store parameters.
template <typename Load, typename Store> class TemplatedIndexedValue;

// By default, edsc::IndexedValue provides an index notation around the affine
// load and stores. edsc::StdIndexedValue provides the standard load/store
// counterpart.
using IndexedValue =
    TemplatedIndexedValue<intrinsics::affine_load, intrinsics::affine_store>;
using StdIndexedValue =
    TemplatedIndexedValue<intrinsics::std_load, intrinsics::std_store>;

// Base class for MemRefView and VectorView.
class View {
public:
  unsigned rank() const { return lbs.size(); }
  ValueHandle lb(unsigned idx) { return lbs[idx]; }
  ValueHandle ub(unsigned idx) { return ubs[idx]; }
  int64_t step(unsigned idx) { return steps[idx]; }
  std::tuple<ValueHandle, ValueHandle, int64_t> range(unsigned idx) {
    return std::make_tuple(lbs[idx], ubs[idx], steps[idx]);
  }
  void swapRanges(unsigned i, unsigned j) {
    if (i == j)
      return;
    lbs[i].swap(lbs[j]);
    ubs[i].swap(ubs[j]);
    std::swap(steps[i], steps[j]);
  }

  ArrayRef<ValueHandle> getLbs() { return lbs; }
  ArrayRef<ValueHandle> getUbs() { return ubs; }
  ArrayRef<int64_t> getSteps() { return steps; }

protected:
  SmallVector<ValueHandle, 8> lbs;
  SmallVector<ValueHandle, 8> ubs;
  SmallVector<int64_t, 8> steps;
};

/// A MemRefView represents the information required to step through a
/// MemRef. It has placeholders for non-contiguous tensors that fit within the
/// Fortran subarray model.
/// At the moment it can only capture a MemRef with an identity layout map.
// TODO(ntv): Support MemRefs with layoutMaps.
class MemRefView : public View {
public:
  explicit MemRefView(Value v);
  MemRefView(const MemRefView &) = default;
  MemRefView &operator=(const MemRefView &) = default;

  unsigned fastestVarying() const { return rank() - 1; }

private:
  friend IndexedValue;
  ValueHandle base;
};

/// A VectorView represents the information required to step through a
/// Vector accessing each scalar element at a time. It is the counterpart of
/// a MemRefView but for vectors. This exists purely for boilerplate avoidance.
class VectorView : public View {
public:
  explicit VectorView(Value v);
  VectorView(const VectorView &) = default;
  VectorView &operator=(const VectorView &) = default;

private:
  friend IndexedValue;
  ValueHandle base;
};

/// A TemplatedIndexedValue brings an index notation over the template Load and
/// Store parameters. This helper class is an abstraction purely for sugaring
/// purposes and allows writing compact expressions such as:
///
/// ```mlir
///    // `IndexedValue` provided by default in the mlir::edsc namespace.
///    using IndexedValue =
///      TemplatedIndexedValue<intrinsics::load, intrinsics::store>;
///    IndexedValue A(...), B(...), C(...);
///    For(ivs, zeros, shapeA, ones, {
///      C(ivs) = A(ivs) + B(ivs)
///    });
/// ```
///
/// Assigning to an IndexedValue emits an actual `Store` operation, while
/// converting an IndexedValue to a ValueHandle emits an actual `Load`
/// operation.
template <typename Load, typename Store> class TemplatedIndexedValue {
public:
  explicit TemplatedIndexedValue(Type t) : base(t) {}
  explicit TemplatedIndexedValue(Value v)
      : TemplatedIndexedValue(ValueHandle(v)) {}
  explicit TemplatedIndexedValue(ValueHandle v) : base(v) {}

  TemplatedIndexedValue(const TemplatedIndexedValue &rhs) = default;

  TemplatedIndexedValue operator()() { return *this; }
  /// Returns a new `TemplatedIndexedValue`.
  TemplatedIndexedValue operator()(ValueHandle index) {
    TemplatedIndexedValue res(base);
    res.indices.push_back(index);
    return res;
  }
  template <typename... Args>
  TemplatedIndexedValue operator()(ValueHandle index, Args... indices) {
    return TemplatedIndexedValue(base, index).append(indices...);
  }
  TemplatedIndexedValue operator()(ArrayRef<ValueHandle> indices) {
    return TemplatedIndexedValue(base, indices);
  }
  TemplatedIndexedValue operator()(ArrayRef<IndexHandle> indices) {
    return TemplatedIndexedValue(
        base, ArrayRef<ValueHandle>(indices.begin(), indices.end()));
  }

  /// Emits a `store`.
  // NOLINTNEXTLINE: unconventional-assign-operator
  OperationHandle operator=(const TemplatedIndexedValue &rhs) {
    ValueHandle rrhs(rhs);
    return Store(rrhs, getBase(), {indices.begin(), indices.end()});
  }
  // NOLINTNEXTLINE: unconventional-assign-operator
  OperationHandle operator=(ValueHandle rhs) {
    return Store(rhs, getBase(), {indices.begin(), indices.end()});
  }

  /// Emits a `load` when converting to a ValueHandle.
  operator ValueHandle() const {
    return Load(getBase(), {indices.begin(), indices.end()});
  }

  /// Emits a `load` when converting to a Value.
  Value operator*(void) const {
    return Load(getBase(), {indices.begin(), indices.end()}).getValue();
  }

  ValueHandle getBase() const { return base; }

  /// Operator overloadings.
  ValueHandle operator+(ValueHandle e);
  ValueHandle operator-(ValueHandle e);
  ValueHandle operator*(ValueHandle e);
  ValueHandle operator/(ValueHandle e);
  OperationHandle operator+=(ValueHandle e);
  OperationHandle operator-=(ValueHandle e);
  OperationHandle operator*=(ValueHandle e);
  OperationHandle operator/=(ValueHandle e);
  ValueHandle operator+(TemplatedIndexedValue e) {
    return *this + static_cast<ValueHandle>(e);
  }
  ValueHandle operator-(TemplatedIndexedValue e) {
    return *this - static_cast<ValueHandle>(e);
  }
  ValueHandle operator*(TemplatedIndexedValue e) {
    return *this * static_cast<ValueHandle>(e);
  }
  ValueHandle operator/(TemplatedIndexedValue e) {
    return *this / static_cast<ValueHandle>(e);
  }
  OperationHandle operator+=(TemplatedIndexedValue e) {
    return this->operator+=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator-=(TemplatedIndexedValue e) {
    return this->operator-=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator*=(TemplatedIndexedValue e) {
    return this->operator*=(static_cast<ValueHandle>(e));
  }
  OperationHandle operator/=(TemplatedIndexedValue e) {
    return this->operator/=(static_cast<ValueHandle>(e));
  }

private:
  TemplatedIndexedValue(ValueHandle base, ArrayRef<ValueHandle> indices)
      : base(base), indices(indices.begin(), indices.end()) {}

  TemplatedIndexedValue &append() { return *this; }

  template <typename T, typename... Args>
  TemplatedIndexedValue &append(T index, Args... indices) {
    this->indices.push_back(static_cast<ValueHandle>(index));
    append(indices...);
    return *this;
  }
  ValueHandle base;
  SmallVector<ValueHandle, 8> indices;
};

/// Operator overloadings.
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator+(ValueHandle e) {
  using op::operator+;
  return static_cast<ValueHandle>(*this) + e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator-(ValueHandle e) {
  using op::operator-;
  return static_cast<ValueHandle>(*this) - e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator*(ValueHandle e) {
  using op::operator*;
  return static_cast<ValueHandle>(*this) * e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator/(ValueHandle e) {
  using op::operator/;
  return static_cast<ValueHandle>(*this) / e;
}

template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator+=(ValueHandle e) {
  using op::operator+;
  return Store(*this + e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator-=(ValueHandle e) {
  using op::operator-;
  return Store(*this - e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator*=(ValueHandle e) {
  using op::operator*;
  return Store(*this * e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator/=(ValueHandle e) {
  using op::operator/;
  return Store(*this / e, getBase(), {indices.begin(), indices.end()});
}

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_HELPERS_H_
