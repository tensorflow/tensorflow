/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_UTIL_H_
#define MLIR_HLO_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_UTIL_H_

#include <complex>
#include <type_traits>
#include <utility>

#include "tools/mlir_interpreter/framework/interpreter_value.h"

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
    if constexpr (Fn::template supportedType<T>()) {
      using out_elem_t = decltype(Fn::apply(T()));
      auto out = TensorOrMemref<out_elem_t>::emptyLike(t.view);
      for (const auto& index : out.view.indices(true)) {
        out.at(index) = Fn::apply(t.at(index));
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
    if constexpr (Fn::template supportedType<T>()) {
      return {Fn::apply(t)};
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
  InterpreterValue operator()(const TensorOrMemref<T>& lhsT) {
    if constexpr (Fn::template supportedType<T>()) {
      using OutElemT = decltype(Fn::apply(T(), T()));
      auto out = TensorOrMemref<OutElemT>::emptyLike(lhsT.view);
      const auto& rhsT = std::get<TensorOrMemref<T>>(rhs.storage);
      for (const auto& index : out.view.indices(true)) {
        out.at(index) = Fn::apply(lhsT.at(index), rhsT.at(index));
      }
      return {out};
    } else {
      llvm::errs() << llvm::getTypeName<Fn>()
                   << " unsupported type: " << llvm::getTypeName<T>() << "\n";
      llvm_unreachable("unsupported type");
    }
  }

  InterpreterValue operator()(const Tuple& lhsT) {
    const auto& rhsT = std::get<Tuple>(rhs.storage);
    Tuple out;
    for (const auto& [lhs_v, rhs_v] : llvm::zip(lhsT.values, rhsT.values)) {
      out.values.push_back(std::make_unique<InterpreterValue>(std::move(
          std::visit(InterpreterValueBiMapVisitor{*rhs_v}, lhs_v->storage))));
    }
    return {std::move(out)};
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    if constexpr (Fn::template supportedType<T>()) {
      return {Fn::apply(t, std::get<T>(rhs.storage))};
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
  static constexpr bool supportedType() {
    constexpr bool isBool = std::is_same_v<T, bool>;
    constexpr bool isInt = std::is_integral_v<T> && !isBool;
    constexpr bool isUnsigned = std::is_unsigned_v<T>;
    return (allow_bools && isBool) ||
           (allow_ints && isInt && (allow_unsigned || !isUnsigned)) ||
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
InterpreterValue applyCwiseMap(const InterpreterValue& value) {
  return std::visit(detail::InterpreterValueMapVisitor<Fn>{}, value.storage);
}

template <typename Fn>
InterpreterValue applyCwiseBinaryMap(const InterpreterValue& lhs,
                                     const InterpreterValue& rhs) {
  assert(lhs.storage.index() == rhs.storage.index());
  return std::visit(detail::InterpreterValueBiMapVisitor<Fn>{rhs}, lhs.storage);
}

// Unboxes (and casts if necessary) the given interpreter values. Asserts if the
// types are incompatible.
template <typename T>
SmallVector<T> unpackInterpreterValues(ArrayRef<InterpreterValue> values) {
  SmallVector<T> result;
  for (const auto& value : values) {
    result.push_back(interpreterValueCast<T>(value));
  }
  return result;
}

// Boxes the given values in InterpreterValues.
template <typename T>
SmallVector<InterpreterValue> packInterpreterValues(ArrayRef<T> values) {
  SmallVector<InterpreterValue> result;
  for (const auto& value : values) {
    result.push_back({value});
  }
  return result;
}

}  // namespace interpreter
}  // namespace mlir

#endif  // MLIR_HLO_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_UTIL_H_
