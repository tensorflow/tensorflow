//===- Functional.h - Helpers for functional-style Combinators --*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_FUNCTIONAL_H_
#define MLIR_SUPPORT_FUNCTIONAL_H_

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

/// This file provides some simple template functional-style sugar to operate
/// on **value** types. Make sure when using that the stored type is cheap to
/// copy!
///
/// TODO(ntv): add some static_assert but we need proper traits for this.

namespace mlir {
namespace functional {

/// Map with iterators.
template <typename Fun, typename IterType>
auto map(Fun fun, IterType begin, IterType end)
    -> llvm::SmallVector<typename std::result_of<Fun(decltype(*begin))>::type,
                         8> {
  using R = typename std::result_of<Fun(decltype(*begin))>::type;
  llvm::SmallVector<R, 8> res;
  // auto i works with both pointer types and value types with an operator*.
  // auto *i only works for pointer types.
  for (auto i = begin; i != end; ++i) {
    res.push_back(fun(*i));
  }
  return res;
}

/// Map with templated container.
template <typename Fun, typename ContainerType>
auto map(Fun fun, ContainerType input)
    -> decltype(map(fun, std::begin(input), std::end(input))) {
  return map(fun, std::begin(input), std::end(input));
}

/// Apply with iterators.
template <typename Fun, typename IterType>
void apply(Fun fun, IterType begin, IterType end) {
  // auto i works with both pointer types and value types with an operator*.
  // auto *i only works for pointer types.
  for (auto i = begin; i != end; ++i) {
    fun(*i);
  }
}

/// Apply with templated container.
template <typename Fun, typename ContainerType>
void apply(Fun fun, ContainerType input) {
  return apply(fun, std::begin(input), std::end(input));
}

/// Zip apply with 2 templated container, iterates to the min of the sizes of
/// the 2 containers.
/// TODO(ntv): make variadic.
template <typename Fun, typename ContainerType1, typename ContainerType2>
void zipApply(Fun fun, ContainerType1 input1, ContainerType2 input2) {
  auto zipIter = llvm::zip(input1, input2);
  for (auto it : zipIter) {
    fun(std::get<0>(it), std::get<1>(it));
  }
}

/// Unwraps a pointer type to another type (possibly the same).
/// Used in particular to allow easier compositions of
///   llvm::iterator_range<ForInst::operand_iterator> types.
template <typename T, typename ToType = T>
inline std::function<ToType *(T *)> makePtrDynCaster() {
  return [](T *val) { return llvm::dyn_cast<ToType>(val); };
}

/// Simple ScopeGuard.
struct ScopeGuard {
  explicit ScopeGuard(std::function<void(void)> destruct)
      : destruct(destruct) {}
  ~ScopeGuard() { destruct(); }

private:
  std::function<void(void)> destruct;
};

} // namespace functional
} // namespace mlir

#endif // MLIR_SUPPORT_FUNCTIONAL_H_
