/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_UTIL_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_UTIL_H_

#include <cassert>
#include <complex>
#include <type_traits>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace detail {

template <typename T>
struct is_complex : std::false_type {};  // NOLINT

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};  // NOLINT

template <typename Fn>
struct InterpreterValueMapVisitor {
  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& t) {
    if constexpr (Fn::template SupportedType<T>()) {
      using out_elem_t = decltype(Fn::Apply(T()));
      auto out = TensorOrMemref<out_elem_t>::EmptyLike(t.view);
      for (const auto& index : out.view.Indices(true)) {
        out.at(index) = Fn::Apply(t.at(index));
      }
      return {out};
    } else {
      llvm::errs() << llvm::getTypeName<Fn>()
                   << " unsupported type: " << llvm::getTypeName<T>() << "\n";
      llvm_unreachable("unsupported type");
    }
  }

  InterpreterValue operator()(const Tuple& t) {
    Tuple out;
    for (const auto& value : t.values) {
      out.values.push_back(std::make_unique<InterpreterValue>(
          std::move(std::visit(*this, value->storage))));
    }
    return {out};
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    if constexpr (Fn::template SupportedType<T>()) {
      return {Fn::Apply(t)};
    } else {
      llvm::errs() << llvm::getTypeName<Fn>()
                   << " unsupported type: " << llvm::getTypeName<T>() << "\n";
      llvm_unreachable("unsupported type");
    }
  }
};

template <typename Fn>
struct InterpreterValueBiMapVisitor {
  const InterpreterValue& rhs;

  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& lhs_t) {
    if constexpr (Fn::template SupportedType<T>()) {
      using OutElemT = decltype(Fn::Apply(T(), T()));
      auto out = TensorOrMemref<OutElemT>::EmptyLike(lhs_t.view);
      const auto& rhs_t = std::get<TensorOrMemref<T>>(rhs.storage);
      for (const auto& index : out.view.Indices(true)) {
        out.at(index) = Fn::Apply(lhs_t.at(index), rhs_t.at(index));
      }
      return {out};
    } else {
      llvm::errs() << llvm::getTypeName<Fn>()
                   << " unsupported type: " << llvm::getTypeName<T>() << "\n";
      llvm_unreachable("unsupported type");
    }
  }

  InterpreterValue operator()(const Tuple& lhs_t) {
    const auto& rhs_t = std::get<Tuple>(rhs.storage);
    Tuple out;
    for (const auto& [lhs_v, rhs_v] : llvm::zip(lhs_t.values, rhs_t.values)) {
      out.values.push_back(std::make_unique<InterpreterValue>(std::move(
          std::visit(InterpreterValueBiMapVisitor{*rhs_v}, lhs_v->storage))));
    }
    return {std::move(out)};
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    if constexpr (Fn::template SupportedType<T>()) {
      return {Fn::Apply(t, std::get<T>(rhs.storage))};
    } else {
      llvm::errs() << llvm::getTypeName<Fn>()
                   << " unsupported type: " << llvm::getTypeName<T>() << "\n";
      llvm_unreachable("unsupported type");
    }
  }
};

}  // namespace detail

template <typename T>
inline constexpr bool is_complex_v = detail::is_complex<T>::value;  // NOLINT

template <bool allow_bools, bool allow_ints, bool allow_floats,
          bool allow_complex, bool allow_unsigned = true>
struct FilterMapTraits {
  template <typename T>
  static constexpr bool SupportedType() {
    constexpr bool is_bool = std::is_same_v<T, bool>;
    constexpr bool is_int = std::is_integral_v<T> && !is_bool;
    constexpr bool is_unsigned = std::is_unsigned_v<T>;
    return (allow_bools && is_bool) ||
           (allow_ints && is_int && (allow_unsigned || !is_unsigned)) ||
           (allow_floats && std::is_floating_point_v<T>) ||
           (allow_complex && is_complex_v<T>);
  }
};

using CwiseAll = FilterMapTraits<true, true, true, true>;
using CwiseArith = FilterMapTraits<false, true, true, true>;
using CwiseComplex = FilterMapTraits<false, false, false, true>;
using CwiseFloat = FilterMapTraits<false, false, true, false>;
using CwiseInt = FilterMapTraits<false, true, false, false>;
using CwiseIntegral = FilterMapTraits<true, true, false, false>;
using CwiseNonIntegral = FilterMapTraits<false, false, true, true>;
using CwiseReal = FilterMapTraits<false, true, true, false>;
using CwiseSignedOrComplex = FilterMapTraits<false, true, true, true, false>;
using CwiseSigned = FilterMapTraits<false, true, true, false, false>;

template <typename Fn>
InterpreterValue ApplyCwiseMap(const InterpreterValue& value) {
  return std::visit(detail::InterpreterValueMapVisitor<Fn>{}, value.storage);
}

template <typename Fn>
InterpreterValue ApplyCwiseBinaryMap(const InterpreterValue& lhs,
                                     const InterpreterValue& rhs) {
  assert(lhs.storage.index() == rhs.storage.index());
  return std::visit(detail::InterpreterValueBiMapVisitor<Fn>{rhs}, lhs.storage);
}

// Unboxes (and casts if necessary) the given interpreter values. Asserts if the
// types are incompatible.
template <typename T>
SmallVector<T> UnpackInterpreterValues(ArrayRef<InterpreterValue> values) {
  SmallVector<T> result;
  for (const auto& value : values) {
    result.push_back(InterpreterValueCast<T>(value));
  }
  return result;
}

// Boxes the given values in InterpreterValues.
template <typename T>
SmallVector<InterpreterValue> PackInterpreterValues(ArrayRef<T> values) {
  SmallVector<InterpreterValue> result;
  for (const auto& value : values) {
    result.push_back({value});
  }
  return result;
}

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_UTIL_H_
