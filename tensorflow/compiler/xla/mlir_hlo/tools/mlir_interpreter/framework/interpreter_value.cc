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

#include "tools/mlir_interpreter/framework/interpreter_value.h"

#include <complex>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

namespace mlir {
namespace interpreter {

namespace {
struct TypeStr {
  static std::string_view get(bool) { return "i1"; }
  static std::string_view get(int64_t) { return "i64"; }
  static std::string_view get(int32_t) { return "i32"; }
  static std::string_view get(int16_t) { return "i16"; }
  static std::string_view get(int8_t) { return "i8"; }
  static std::string_view get(uint64_t) { return "ui64"; }
  static std::string_view get(uint32_t) { return "ui32"; }
  static std::string_view get(uint16_t) { return "ui16"; }
  static std::string_view get(uint8_t) { return "ui8"; }
  static std::string_view get(float) { return "f32"; }
  static std::string_view get(double) { return "f64"; }
  static std::string_view get(std::complex<float>) { return "complex<f32>"; }
  static std::string_view get(std::complex<double>) { return "complex<f64>"; }
};

struct InterpreterValuePrinter {
  llvm::raw_ostream& os;

  template <typename T>
  void operator()(const TensorOrMemref<T>& t) {
    if (!t.buffer) {
      os << "Memref: null";
      return;
    }

    if (t.view.isVector) {
      os << "vector<";
    } else {
      os << "TensorOrMemref<";
    }
    ArrayRef<int64_t> sizes = t.view.sizes;
    for (int64_t size : sizes.drop_back(t.view.numVectorDims.value_or(0))) {
      os << size << "x";
    }
    if (t.view.numVectorDims) {
      os << "vector<";
      for (int64_t size : sizes.take_back(*t.view.numVectorDims)) {
        os << size << "x";
      }
      os << TypeStr::get(T{}) << ">>: ";
    } else {
      os << TypeStr::get(T{}) << ">: ";
    }
    SmallVector<int64_t> indices(t.view.rank() +
                                 t.view.numVectorDims.value_or(0));
    std::function<void(int64_t)> print;
    print = [&](int64_t dim) {
      if (dim == indices.size()) {
        printScalar(t.at(indices));
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
    if (t.buffer->deallocated()) {
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
      v->print(os);
    }
    os << ")";
  }

  template <typename T>
  void operator()(const T& t) {
    os << TypeStr::get(t) << ": ";
    printScalar(t);
  }

  template <typename T>
  void printScalar(const T& v) {
    os << v;
  }

  template <typename T>
  void printScalar(const std::complex<T>& v) {
    os << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "i";
  }

  void printScalar(bool v) { os << (v ? "true" : "false"); }

  void printScalar(int8_t v) { os << (int)v; }
  void printScalar(uint8_t v) { os << (int)v; }
};
}  // namespace

void InterpreterValue::print(llvm::raw_ostream& os) const {
  std::visit(InterpreterValuePrinter{os}, storage);
}

std::string InterpreterValue::toString() const {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  print(os);
  return buf;
}

InterpreterValue InterpreterValue::extractElement(
    llvm::ArrayRef<int64_t> indices) const {
  return std::visit(
      [&](auto& it) -> InterpreterValue {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          if (it.view.numVectorDims) {
            return {it.vectorAt(indices)};
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

void InterpreterValue::insertElement(llvm::ArrayRef<int64_t> indices,
                                     const InterpreterValue& value) {
  std::visit(
      [&](auto& it) {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          if (it.view.numVectorDims) {
            auto subview = it.vectorAt(indices);
            const auto& values = std::get<T>(value.storage);
            assert(values.view.sizes == subview.view.sizes &&
                   "mismatched sizes");
            for (const auto& index : subview.view.indices()) {
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

void InterpreterValue::fill(
    const std::function<InterpreterValue(llvm::ArrayRef<int64_t> indices)>& f) {
  std::visit(
      [&](auto& it) {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          for (const auto& indices : it.view.indices()) {
            if (it.view.numVectorDims) {
              auto subview = it.vectorAt(indices);
              auto value = std::get<T>(f(indices).storage);
              for (const auto& index : subview.view.indices()) {
                subview.at(index) = value.at(index);
              }
            } else {
              it.at(indices) =
                  std::get<typename T::element_type>(f(indices).storage);
            }
          }
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("filling tuples is unsupported");
        } else {
          it = std::get<T>(f({}).storage);
        }
      },
      storage);
}

InterpreterValue InterpreterValue::clone(ArrayRef<int64_t> layout) const {
  return std::visit(
      [&](const auto& it) -> InterpreterValue {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          return {it.clone(layout)};
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("cloning tuples is unsupported");
        } else {
          return {it};
        }
      },
      storage);
}

InterpreterValue InterpreterValue::coerceLayout(
    ArrayRef<int64_t> layout) const {
  const auto& view = this->view();
  if (view.strides == BufferView::getStridesForLayout(view.sizes, layout)) {
    return *this;
  }
  return clone(layout);
}

InterpreterValue InterpreterValue::typedAlike(
    llvm::ArrayRef<int64_t> shape) const {
  return std::visit(
      [&](const auto& it) -> InterpreterValue {
        using T = std::decay_t<decltype(it)>;
        if constexpr (is_tensor_or_memref_v<T>) {
          return {T::empty(shape)};
        } else if constexpr (std::is_same_v<T, Tuple>) {
          llvm_unreachable("TypedAlike for tuples is unsupported");
        } else {
          return {TensorOrMemref<T>::empty(shape)};
        }
      },
      storage);
}

InterpreterValue InterpreterValue::makeTensor(mlir::Type elementType,
                                              SmallVector<int64_t> shape) {
  auto vectorTy = llvm::dyn_cast<VectorType>(elementType);
  if (vectorTy) {
    llvm::copy(vectorTy.getShape(), std::back_inserter(shape));
  }
  return dispatchScalarType(elementType, [&](auto dummy) -> InterpreterValue {
    auto tensor = TensorOrMemref<decltype(dummy)>::empty(shape);
    if (vectorTy) {
      tensor.view.numVectorDims = vectorTy.getRank();
    }
    return {tensor};
  });
}

BufferView& InterpreterValue::view() {
  return std::visit(
      [](auto& it) -> BufferView& {
        if constexpr (is_tensor_or_memref_v<decltype(it)>) {
          return it.view;
        }
        llvm_unreachable("view is only supported for tensors");
      },
      storage);
}

const BufferView& InterpreterValue::view() const {
  return std::visit(
      [](const auto& it) -> const BufferView& {
        if constexpr (is_tensor_or_memref_v<decltype(it)>) {
          return it.view;
        }
        llvm_unreachable("view is only supported for tensors");
      },
      storage);
}

bool InterpreterValue::isTensor() const {
  return std::visit(
      [](const auto& it) { return is_tensor_or_memref_v<decltype(it)>; },
      storage);
}

InterpreterValue InterpreterValue::asUnitTensor(bool isVector) const {
  auto result = typedAlike({});
  result.insertElement({}, *this);
  result.view().isVector = isVector;
  return result;
}

bool Tuple::operator==(const Tuple& other) const {
  if (other.values.size() != values.size()) return false;
  for (const auto& [lhs, rhs] : llvm::zip(values, other.values)) {
    if (!(*lhs == *rhs)) return false;
  }
  return true;
}

std::shared_ptr<Buffer> InterpreterValue::buffer() const {
  return std::visit(
      [](const auto& it) -> std::shared_ptr<Buffer> {
        if constexpr (is_tensor_or_memref_v<decltype(it)>) {
          return it.buffer;
        } else {
          llvm_unreachable("buffer() is only supported for tensors");
        }
      },
      storage);
}

int64_t InterpreterValue::asInt() const {
  auto visit = [](auto value) -> int64_t {
    if constexpr (std::is_integral_v<decltype(value)>) {
      return static_cast<int64_t>(value);
    } else {
      llvm_unreachable("only integral types can be converted to ints");
    }
  };
  return std::visit(visit, storage);
}

uint64_t InterpreterValue::asUInt() const {
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

double InterpreterValue::asDouble() const {
  auto visit = [](auto value) -> int64_t {
    if constexpr (std::is_floating_point_v<decltype(value)>) {
      return static_cast<double>(value);
    } else {
      llvm_unreachable("only float types can be converted to ints");
    }
  };
  return std::visit(visit, storage);
}

int64_t InterpreterValue::getByteSizeOfElement() const {
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
