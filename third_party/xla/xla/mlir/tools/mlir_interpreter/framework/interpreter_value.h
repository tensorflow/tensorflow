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

#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_H_

#include <cassert>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {

struct InterpreterValue;

struct Tuple {
  bool operator==(const Tuple& other) const;

  SmallVector<std::shared_ptr<InterpreterValue>> values;
};

// Holds a scalar, a tensor/memref or a tuple. Tensors/memrefs can also
// represent vectors.
struct InterpreterValue {
  void Print(llvm::raw_ostream& os) const;
  std::string ToString() const;

  // Returns the element at the given indices. If the value is a scalar, returns
  // itself.
  InterpreterValue ExtractElement(llvm::ArrayRef<int64_t> indices) const;
  // Sets the element at the given index. If the value is a scalar, sets its
  // value.
  void InsertElement(llvm::ArrayRef<int64_t> indices,
                     const InterpreterValue& value);
  // Initializes all elements of the underlying tensor.
  void Fill(
      const std::function<InterpreterValue(llvm::ArrayRef<int64_t> indices)>&
          f);

  // Converts a scalar to a unit tensor or vector.
  InterpreterValue AsUnitTensor(bool is_vector = false) const;
  // For integral interpreter values, casts them to int64.
  int64_t AsInt() const;
  // For integral interpreter values, first casts them to the unsigned integer
  // type of the same size, and then to uint64. For example, the result for
  // int8_t{-1} is 255.
  uint64_t AsUInt() const;
  // For floating point scalars, casts them to double.
  double AsDouble() const;
  // Must be a tensor or memref.
  int64_t GetByteSizeOfElement() const;

  // Creates a new tensor InterpreterValue (backed a new buffer) with the same
  // element type as this, but a different shape. If this is not a tensor, it is
  // used as the element type.
  // If `layout` is empty, the clone uses the default layout.
  InterpreterValue Clone(ArrayRef<int64_t> layout = {}) const;
  // Returns either this tensor InterpreterValue (if its layout matches the
  // requested layout) or a clone.
  InterpreterValue CoerceLayout(llvm::ArrayRef<int64_t> layout) const;
  // Returns a tensor interpreter value with a newly allocated buffer of the
  // given shape, with a default layout and the same element type as this
  // interpreter value.
  InterpreterValue TypedAlike(llvm::ArrayRef<int64_t> shape) const;

  // Creates a tensor with the given element type and shape. `element_type` may
  // be a vector type, in which case the shape only specifies the non-vector
  // dimensions.
  static InterpreterValue MakeTensor(mlir::Type element_type,
                                     SmallVector<int64_t> shape);

  // Returns the underlying tensor's view. Must be a tensor.
  BufferView& View();
  const BufferView& View() const;
  // Returns the underlying tensor's buffer. Must be a tensor.
  std::shared_ptr<Buffer> GetBuffer() const;

  bool IsTensor() const;

  bool operator==(const InterpreterValue& other) const {
    if (storage.index() != other.storage.index()) return false;
    if (IsTensor() || std::holds_alternative<Tuple>(storage))
      return storage == other.storage;
    // Tensors treat NaNs as equal, so just wrap the values.
    return AsUnitTensor() == other.AsUnitTensor();
  }

  std::variant<
      Tuple, bool, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t,
      int32_t, uint64_t, int64_t, std::complex<float>, std::complex<double>,
      TensorOrMemref<bool>, TensorOrMemref<float>, TensorOrMemref<double>,
      TensorOrMemref<uint8_t>, TensorOrMemref<int8_t>, TensorOrMemref<uint16_t>,
      TensorOrMemref<int16_t>, TensorOrMemref<uint32_t>,
      TensorOrMemref<int32_t>, TensorOrMemref<uint64_t>,
      TensorOrMemref<int64_t>, TensorOrMemref<std::complex<float>>,
      TensorOrMemref<std::complex<double>>>
      storage;
};

template <typename T>
constexpr static bool is_valid_interpreter_value_v =  // NOLINT
    std::is_constructible_v<decltype(InterpreterValue::storage), T>;

// Attempts to cast the given value to the requested type, returning nullopt if
// no cast is possible. This allows casts to the concrete type of the value
// (e.g. an `InterpreterValue` containing a `Tuple` can be cast to `Tuple`),
// casts from a unit tensor to their contents, and casts of scalars to any
// convertible type.
// NOTE: When casting to an unsigned type, this behaves differently than
// InterpreterValue::AsUint. That function preserves the content's bit width,
// so InterpreterValueDynCast<uint64_t>({int8_t{-1}}) will return 2^64-1,
// whereas AsUInt will return 255.
template <typename T>
std::optional<T> InterpreterValueDynCast(InterpreterValue v) {
  if constexpr (std::is_same_v<T, InterpreterValue>) {
    return v;
  }
  if constexpr (is_valid_interpreter_value_v<T>) {
    if (std::holds_alternative<T>(v.storage)) {
      return std::get<T>(v.storage);
    }
  }
  if (v.IsTensor() && !is_tensor_or_memref_v<T>) {
    if (v.View().GetNumElements() != 1) {
      return std::nullopt;
    }
    return InterpreterValueDynCast<T>(v.ExtractElement({}));
  }
  return std::visit(
      [](auto v) -> std::optional<T> {
        if constexpr (std::is_convertible_v<decltype(v), T>) {
          return v;
        } else {
          return std::nullopt;
        }
      },
      v.storage);
}

template <typename T>
T InterpreterValueCast(InterpreterValue v) {
  auto ret = InterpreterValueDynCast<T>(v);
  assert(ret && "cast failed");
  return *std::move(ret);
}

// Calls functor with a value of the C++ type corresponding to the given `Type`,
// (or its element type).
template <class Fn>
auto DispatchScalarType(mlir::Type ty, Fn&& functor) {
  ty = getElementTypeOrSelf(ty);
  if (ty.isF32()) {
    return functor(float{});
  }
  if (ty.isF64()) {
    return functor(double{});
  }
  if (ty.isUnsignedInteger(64)) {
    return functor(uint64_t{});
  }
  if (ty.isInteger(64) || ty.isIndex()) {
    return functor(int64_t{});
  }
  if (ty.isUnsignedInteger(32)) {
    return functor(uint32_t{});
  }
  if (ty.isInteger(32)) {
    return functor(int32_t{});
  }
  if (ty.isUnsignedInteger(16)) {
    return functor(uint16_t{});
  }
  if (ty.isInteger(16)) {
    return functor(int16_t{});
  }
  if (ty.isUnsignedInteger(8)) {
    return functor(uint8_t{});
  }
  if (ty.isInteger(8)) {
    return functor(int8_t{});
  }
  if (ty.isInteger(1)) {
    return functor(bool{});
  }
  if (auto complex = mlir::dyn_cast<ComplexType>(ty)) {
    if (complex.getElementType().isF32()) {
      return functor(std::complex<float>{});
    }
    if (complex.getElementType().isF64()) {
      return functor(std::complex<double>{});
    }
  }

  llvm::errs() << "DispatchScalarType unimplemented for " << ty << "\n";
  llvm_unreachable("unimplemented");
}

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_INTERPRETER_VALUE_H_
