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

#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"

#include <cassert>
#include <complex>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>

#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {

namespace {
struct TypeStr {
  static absl::string_view Get(bool) { return "i1"; }
  static absl::string_view Get(int64_t) { return "i64"; }
  static absl::string_view Get(int32_t) { return "i32"; }
  static absl::string_view Get(int16_t) { return "i16"; }
  static absl::string_view Get(int8_t) { return "i8"; }
  static absl::string_view Get(uint64_t) { return "ui64"; }
  static absl::string_view Get(uint32_t) { return "ui32"; }
  static absl::string_view Get(uint16_t) { return "ui16"; }
  static absl::string_view Get(uint8_t) { return "ui8"; }
  static absl::string_view Get(float) { return "f32"; }
  static absl::string_view Get(double) { return "f64"; }
  static absl::string_view Get(std::complex<float>) { return "complex<f32>"; }
  static absl::string_view Get(std::complex<double>) { return "complex<f64>"; }
};

struct InterpreterValuePrinter {
  llvm::raw_ostream& os;

  template <typename T>
  void operator()(const TensorOrMemref<T>& t) {
    if (!t.buffer) {
      os << "Memref: null";
      return;
    }

    if (t.view.is_vector) {
      os << "vector<";
    } else {
      os << "TensorOrMemref<";
    }
    ArrayRef<int64_t> sizes = t.view.sizes;
    for (int64_t size : sizes.drop_back(t.view.num_vector_dims.value_or(0))) {
      os << size << "x";
    }
    if (t.view.num_vector_dims) {
      os << "vector<";
      for (int64_t size : sizes.take_back(*t.view.num_vector_dims)) {
        os << size << "x";
      }
      os << TypeStr::Get(T{}) << ">>: ";
    } else {
      os << TypeStr::Get(T{}) << ">: ";
    }
    SmallVector<int64_t> indices(t.view.num_dimensions() +
                                 t.view.num_vector_dims.value_or(0));
    std::function<void(int64_t)> print;
    print = [&](int64_t dim) {
      if (dim == indices.size()) {
        PrintScalar(t.at(indices));
      } else {
        os << "[";
        for (int64_t i = 0; i < t.view.sizes[dim]; ++i) {
          if (i > 0) os << ", ";
          indices[dim] = i;
          print(dim + 1);
        }
        os << "]";
      }
    };
    if (t.buffer->Deallocated()) {
      os << "<<deallocated>>";
    } else {
      print(0);
    }
  }

  void operator()(const Tuple& t) {
    os << "(";
    bool first = true;
    for (const auto& v : t.values) {
      if (!first) os << ", ";
      first = false;
      v->Print(os);
    }
    os << ")";
  }

  template <typename T>
  void operator()(const T& t) {
    os << TypeStr::Get(t) << ": ";
    PrintScalar(t);
  }

  template <typename T>
  void PrintScalar(const T& v) {
    os << v;
  }

  template <typename T>
  void PrintScalar(const std::complex<T>& v) {
    os << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "i";
  }

  void PrintScalar(bool v) { os << (v ? "true" : "false"); }

  void PrintScalar(int8_t v) { os << (int)v; }
  void PrintScalar(uint8_t v) { os << (int)v; }
};
}  // namespace

void InterpreterValue::Print(llvm::raw_ostream& os) const {
  std::visit(InterpreterValuePrinter{os}, storage);
}

std::string InterpreterValue::ToString() const {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  Print(os);
  return buf;
}

InterpreterValue InterpreterValue::ExtractElement(
    llvm::ArrayRef<int64_t> indices) const {
  return std::visit(
      [&](auto& it) -> InterpreterValue {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          if (it.view.num_vector_dims) {
            return {it.VectorAt(indices)};
          } else {
            return {it.at(indices)};
          }
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("extracting from tuples is unsupported");
        } else {
          return {it};
        }
      },
      storage);
}

void InterpreterValue::InsertElement(llvm::ArrayRef<int64_t> indices,
                                     const InterpreterValue& value) {
  std::visit(
      [&](auto& it) {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          if (it.view.num_vector_dims) {
            auto subview = it.VectorAt(indices);
            const auto& values = std::get<T>(value.storage);
            assert(values.view.sizes == subview.view.sizes &&
                   "mismatched sizes");
            for (const auto& index : subview.view.Indices()) {
              subview.at(index) = values.at(index);
            }
          } else {
            it.at(indices) = std::get<typename T::element_type>(value.storage);
          }
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("inserting into tuples is unsupported");
        } else {
          it = std::get<T>(value.storage);
        }
      },
      storage);
}

void InterpreterValue::Fill(
    const std::function<InterpreterValue(llvm::ArrayRef<int64_t> indices)>& f) {
  std::visit(
      [&](auto& it) {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          for (const auto& indices : it.view.Indices()) {
            if (it.view.num_vector_dims) {
              auto subview = it.VectorAt(indices);
              auto value = std::get<T>(f(indices).storage);
              for (const auto& index : subview.view.Indices()) {
                subview.at(index) = value.at(index);
              }
            } else {
              it.at(indices) =
                  std::get<typename T::element_type>(f(indices).storage);
            }
          }
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("Filling tuples is unsupported");
        } else {
          it = std::get<T>(f({}).storage);
        }
      },
      storage);
}

InterpreterValue InterpreterValue::Clone(ArrayRef<int64_t> layout) const {
  return std::visit(
      [&](const auto& it) -> InterpreterValue {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          return {it.Clone(layout)};
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("cloning tuples is unsupported");
        } else {
          return {it};
        }
      },
      storage);
}

InterpreterValue InterpreterValue::CoerceLayout(
    ArrayRef<int64_t> layout) const {
  const auto& view = this->View();
  if (view.strides == BufferView::GetStridesForLayout(view.sizes, layout)) {
    return *this;
  }
  return Clone(layout);
}

InterpreterValue InterpreterValue::TypedAlike(
    llvm::ArrayRef<int64_t> shape) const {
  return std::visit(
      [&](const auto& it) -> InterpreterValue {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          return {T::Empty(shape)};
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("TypedAlike for tuples is unsupported");
        } else {
          return {TensorOrMemref<T>::Empty(shape)};
        }
      },
      storage);
}

InterpreterValue InterpreterValue::MakeTensor(mlir::Type element_type,
                                              SmallVector<int64_t> shape) {
  auto vector_ty = llvm::dyn_cast<VectorType>(element_type);
  if (vector_ty) {
    llvm::copy(vector_ty.getShape(), std::back_inserter(shape));
  }
  return DispatchScalarType(element_type, [&](auto dummy) -> InterpreterValue {
    auto tensor = TensorOrMemref<decltype(dummy)>::Empty(shape);
    if (vector_ty) {
      tensor.view.num_vector_dims = vector_ty.getRank();
    }
    return {tensor};
  });
}

BufferView& InterpreterValue::View() {
  return std::visit(
      [](auto& it) -> BufferView& {
        if constexpr (is_tensor_or_memref_v<decltype(it)>) {
          return it.view;
        }
        llvm_unreachable("view is only supported for tensors");
      },
      storage);
}

const BufferView& InterpreterValue::View() const {
  return std::visit(
      [](const auto& it) -> const BufferView& {
        if constexpr (is_tensor_or_memref_v<decltype(it)>) {
          return it.view;
        }
        llvm_unreachable("view is only supported for tensors");
      },
      storage);
}

bool InterpreterValue::IsTensor() const {
  return std::visit(
      [](const auto& it) { return is_tensor_or_memref_v<decltype(it)>; },
      storage);
}

InterpreterValue InterpreterValue::AsUnitTensor(bool is_vector) const {
  auto result = TypedAlike({});
  result.InsertElement({}, *this);
  result.View().is_vector = is_vector;
  return result;
}

bool Tuple::operator==(const Tuple& other) const {
  if (other.values.size() != values.size()) return false;
  for (const auto& [lhs, rhs] : llvm::zip(values, other.values)) {
    if (!(*lhs == *rhs)) return false;
  }
  return true;
}

std::shared_ptr<Buffer> InterpreterValue::GetBuffer() const {
  return std::visit(
      [](const auto& it) -> std::shared_ptr<interpreter::Buffer> {
        if constexpr (is_tensor_or_memref_v<decltype(it)>) {
          return it.buffer;
        } else {
          llvm_unreachable("buffer() is only supported for tensors");
        }
      },
      storage);
}

int64_t InterpreterValue::AsInt() const {
  auto visit = [](auto value) -> int64_t {
    if constexpr (std::is_integral_v<decltype(value)>) {
      return static_cast<int64_t>(value);
    } else {
      llvm_unreachable("only integral types can be converted to ints");
    }
  };
  return std::visit(visit, storage);
}

uint64_t InterpreterValue::AsUInt() const {
  auto visit = [](auto value) -> uint64_t {
    if constexpr (std::is_integral_v<decltype(value)>) {
      if constexpr (std::is_signed_v<decltype(value)>) {
        return static_cast<uint64_t>(
            static_cast<std::make_unsigned_t<decltype(value)>>(value));
      } else {
        return static_cast<uint64_t>(value);
      }
    } else {
      llvm_unreachable("only integral types can be converted to ints");
    }
  };
  return std::visit(visit, storage);
}

double InterpreterValue::AsDouble() const {
  auto visit = [](auto value) -> int64_t {
    if constexpr (std::is_floating_point_v<decltype(value)>) {
      return static_cast<double>(value);
    } else {
      llvm_unreachable("only float types can be converted to ints");
    }
  };
  return std::visit(visit, storage);
}

int64_t InterpreterValue::GetByteSizeOfElement() const {
  return std::visit(
      [](const auto& it) -> int64_t {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          return sizeof(typename T::element_type);
        } else {
          llvm_unreachable("scalars have no element sizes");
        }
      },
      storage);
}

}  // namespace interpreter
}  // namespace mlir
