//===- Rules.h - Helpers for declaring facts and rules ----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helper classes and functions for managing state (facts),
// merging and tracking modification for various data types important for
// quantization.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_RULES_H
#define MLIR_QUANTIZER_SUPPORT_RULES_H

#include "llvm/ADT/Optional.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace mlir {
namespace quantizer {

/// Typed indicator of whether a mutator produces a modification.
struct ModificationResult {
  enum ModificationEnum { Retained, Modified } value;
  ModificationResult(ModificationEnum v) : value(v) {}

  ModificationResult operator|(ModificationResult other) {
    if (value == Modified || other.value == Modified) {
      return ModificationResult(Modified);
    } else {
      return ModificationResult(Retained);
    }
  }

  ModificationResult operator|=(ModificationResult other) {
    value =
        (value == Modified || other.value == Modified) ? Modified : Retained;
    return *this;
  }
};

inline ModificationResult modify(bool isModified = true) {
  return ModificationResult{isModified ? ModificationResult::Modified
                                       : ModificationResult::Retained};
}

inline bool modified(ModificationResult m) {
  return m.value == ModificationResult::Modified;
}

/// A fact that can converge through forward propagation alone without the
/// need to track ownership or individual assertions. In practice, this works
/// for static assertions that are either minimized or maximized and do not
/// vary dynamically.
///
/// It is expected that ValueTy is appropriate to pass by value and has an
/// operator==. The BinaryReducer type should have two static methods:
///   using ValueTy : Type of the value.
///   ValueTy initialValue() : Returns the initial value of the fact.
///   ValueTy reduce(ValueTy lhs, ValueTy rhs) : Reduces two values.
template <typename BinaryReducer>
class BasePropagatedFact {
public:
  using ValueTy = typename BinaryReducer::ValueTy;
  using ThisTy = BasePropagatedFact<BinaryReducer>;
  BasePropagatedFact()
      : value(BinaryReducer::initialValue()),
        salience(std::numeric_limits<int>::min()) {}

  int getSalience() const { return salience; }
  bool hasValue() const { return salience != std::numeric_limits<int>::min(); }
  ValueTy getValue() const { return value; }
  ModificationResult assertValue(int assertSalience, ValueTy assertValue) {
    if (assertSalience > salience) {
      // New salience band.
      value = assertValue;
      salience = assertSalience;
      return modify(true);
    } else if (assertSalience < salience) {
      // Lower salience - ignore.
      return modify(false);
    }
    // Merge within same salience band.
    ValueTy updatedValue = BinaryReducer::reduce(value, assertValue);
    auto mod = modify(value != updatedValue);
    value = updatedValue;
    return mod;
  }
  ModificationResult mergeFrom(const ThisTy &other) {
    if (other.hasValue()) {
      return assertValue(other.getSalience(), other.getValue());
    }
    return modify(false);
  }

private:
  ValueTy value;
  int salience;
};

/// A binary reducer that expands a min/max range represented by a pair
/// of doubles such that it represents the largest of all inputs.
/// The initial value is (Inf, -Inf).
struct ExpandingMinMaxReducer {
  using ValueTy = std::pair<double, double>;
  static ValueTy initialValue() {
    return std::make_pair(std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity());
  }
  static ValueTy reduce(ValueTy lhs, ValueTy rhs) {
    return std::make_pair(std::min(lhs.first, rhs.first),
                          std::max(lhs.second, rhs.second));
  }
};
using ExpandingMinMaxFact = BasePropagatedFact<ExpandingMinMaxReducer>;

/// A binary reducer that minimizing a numeric type.
template <typename T>
struct MinimizingNumericReducer {
  using ValueTy = T;
  static ValueTy initialValue() {
    if (std::numeric_limits<T>::has_infinity()) {
      return std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::max();
    }
  }
  static ValueTy reduce(ValueTy lhs, ValueTy rhs) { return std::min(lhs, rhs); }
};
using MinimizingDoubleFact =
    BasePropagatedFact<MinimizingNumericReducer<double>>;
using MinimizingIntFact = BasePropagatedFact<MinimizingNumericReducer<int>>;

/// A binary reducer that maximizes a numeric type.
template <typename T>
struct MaximizingNumericReducer {
  using ValueTy = T;
  static ValueTy initialValue() {
    if (std::numeric_limits<T>::has_infinity()) {
      return -std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::min();
    }
  }
  static ValueTy reduce(ValueTy lhs, ValueTy rhs) { return std::max(lhs, rhs); }
};
using MaximizingDoubleFact =
    BasePropagatedFact<MaximizingNumericReducer<double>>;
using MaximizingIntFact = BasePropagatedFact<MaximizingNumericReducer<int>>;

/// A fact and reducer for tracking agreement of discrete values. The value
/// type consists of a |T| value and a flag indicating whether there is a
/// conflict (in which case, the preserved value is arbitrary).
template <typename T>
struct DiscreteReducer {
  struct ValueTy {
    ValueTy() : conflict(false) {}
    ValueTy(T value) : value(value), conflict(false) {}
    ValueTy(T value, bool conflict) : value(value), conflict(conflict) {}
    llvm::Optional<T> value;
    bool conflict;
    bool operator==(const ValueTy &other) const {
      if (conflict != other.conflict)
        return false;
      if (value && other.value) {
        return *value == *other.value;
      } else {
        return !value && !other.value;
      }
    }
    bool operator!=(const ValueTy &other) const { return !(*this == other); }
  };
  static ValueTy initialValue() { return ValueTy(); }
  static ValueTy reduce(ValueTy lhs, ValueTy rhs) {
    if (!lhs.value && !rhs.value)
      return lhs;
    else if (!lhs.value)
      return rhs;
    else if (!rhs.value)
      return lhs;
    else
      return ValueTy(*lhs.value, *lhs.value != *rhs.value);
  }
};

template <typename T>
using DiscreteFact = BasePropagatedFact<DiscreteReducer<T>>;

/// Discrete scale/zeroPoint fact.
using DiscreteScaleZeroPointFact = DiscreteFact<std::pair<double, int64_t>>;

} // end namespace quantizer
} // end namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_RULES_H
