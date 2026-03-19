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

#include "xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"

#include <cassert>
#include <complex>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/literal.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace interpreter {
namespace {

// Visitor for converting an InterpreterValue to a TracedValue.
struct TraceInterpreterValueVisitor {
  TracedValue out;

  void Add(float v) { out.add_floats(v); }
  void Add(double v) { out.add_doubles(v); }
  void Add(std::complex<float> v) {
    out.add_floats(v.real());
    out.add_floats(v.imag());
  }
  void Add(std::complex<double> v) {
    out.add_doubles(v.real());
    out.add_doubles(v.imag());
  }
  void Add(int64_t v) { out.add_ints(v); }
  void Add(int32_t v) { out.add_ints(v); }
  void Add(int16_t v) { out.add_ints(v); }
  void Add(int8_t v) { out.add_ints(v); }
  void Add(uint64_t v) { out.add_uints(v); }
  void Add(uint32_t v) { out.add_uints(v); }
  void Add(uint16_t v) { out.add_uints(v); }
  void Add(uint8_t v) { out.add_uints(v); }
  void Add(bool v) { out.add_ints(static_cast<int64_t>(v)); }

  template <typename T>
  void operator()(T v) {
    SetElementType<T>();
    out.set_is_scalar(true);
    Add(v);
  }

  void operator()(const Tuple& t) {
    out.set_element_type(TracedValue::TUPLE);
    for (const auto& v : t.values) {
      *out.add_tuple_elements() = ValueToTracedValue(*v);
    }
  }

  template <typename T>
  void operator()(const TensorOrMemref<T>& v) {
    for (int64_t size : v.view.sizes) {
      out.add_shape(size);
    }
    SetElementType<T>();
    for (const auto& index : v.view.Indices()) {
      Add(v.at(index));
    }
  }

  template <typename T>
  void SetElementType() {
    out.set_element_type(GetElementType(T{}));
    if constexpr (std::is_same_v<T, bool>) {
      out.set_bit_width(1);
    } else {
      out.set_bit_width(sizeof(T) * 8);
    }
  }

  template <typename T>
  static TracedValue::ElementType GetElementType(const T&) {
    if constexpr (std::is_floating_point_v<T>) {
      return TracedValue::FLOAT;
    } else if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_unsigned_v<T>) {
        return TracedValue::UNSIGNED;
      } else {
        return TracedValue::INTEGRAL;
      }
    } else {
      T{"invalid type"} + 0;
      return TracedValue::UNKNOWN;
    }
  }

  template <typename T>
  static TracedValue::ElementType GetElementType(const std::complex<T>&) {
    return TracedValue::COMPLEX;
  }
};

}  // namespace

void ExecutionTraceListener::BeforeOp(ArrayRef<InterpreterValue> args,
                                      Operation* op) {
  auto* inst = regions_.back()->add_instructions();
  inst->set_name(op->getName().getStringRef().str());
  for (const auto& arg : args) {
    *inst->add_args() = ValueToTracedValue(arg);
  }
}

void ExecutionTraceListener::AfterOp(ArrayRef<InterpreterValue> results) {
  auto* traced_results =
      regions_.back()->mutable_instructions()->rbegin()->mutable_results();
  for (const auto& result : results) {
    *traced_results->Add() = ValueToTracedValue(result);
  }
}

void ExecutionTraceListener::EnterRegion(ArrayRef<InterpreterValue> bbargs,
                                         Region& region) {
  if (regions_.empty()) {
    regions_.push_back(trace_->mutable_trace());
  } else {
    regions_.push_back(
        regions_.back()->mutable_instructions()->rbegin()->add_regions());
  }

  auto& traced_region = *regions_.back();
  traced_region.set_region_number(region.getRegionNumber());
  for (const auto& bbarg : bbargs) {
    *traced_region.add_bbargs() = ValueToTracedValue(bbarg);
  }
}

void ExecutionTraceListener::LeaveRegion(ArrayRef<InterpreterValue> yielded) {
  for (const auto& result : yielded) {
    *regions_.back()->add_results() = ValueToTracedValue(result);
  }
  regions_.pop_back();
}

llvm::SmallVector<mlir::Attribute> ValueToAttribute(
    const InterpreterValue& value, mlir::Type type) {
  if (std::holds_alternative<Tuple>(value.storage)) {
    auto types = mlir::cast<TupleType>(type).getTypes();
    const auto& t = std::get<Tuple>(value.storage);
    llvm::SmallVector<mlir::Attribute> attrs;
    for (const auto& [v, ty] : llvm::zip(t.values, types)) {
      auto attr = ValueToAttribute(*v, ty);
      assert(attr.size() == 1 && "nested tuples not supported");
      attrs.push_back(attr.front());
    }
    return attrs;
  }

  if (!value.IsTensor()) {
    return {cast<DenseElementsAttr>(
                ValueToAttribute(value.AsUnitTensor(),
                                 mlir::RankedTensorType::get({}, type))
                    .front())
                .getValues<mlir::Attribute>()[0]};
  }

  if (!mlir::isa<ShapedType>(type)) {
    return {};
  }

  auto shaped_ty = mlir::cast<ShapedType>(type);
  return {DispatchScalarType(shaped_ty, [&](auto dummy) -> mlir::Attribute {
    using T = decltype(dummy);
    auto& t = std::get<TensorOrMemref<T>>(value.storage);
    SmallVector<T> vals;
    for (const auto& index : t.view.Indices()) {
      vals.push_back(t.at(index));
    }
    auto attr_ty =
        shaped_ty.cloneWith(/*shape=*/t.view.sizes, shaped_ty.getElementType());
    if constexpr (std::is_same_v<T, bool>) {
      return mlir::DenseElementsAttr::get(attr_ty, vals);
    } else {
      return mlir::DenseElementsAttr::get<T>(attr_ty, vals);
    }
  })};
}

namespace {
template <typename T>
TensorOrMemref<T> ArrayLiteralToTensor(const xla::Literal& literal) {
  SmallVector<int64_t> layout;
  if (literal.shape().has_layout()) {
    llvm::copy(literal.shape().layout().minor_to_major(),
               std::back_inserter(layout));
  }
  SmallVector<int64_t> shape{literal.shape().dimensions().begin(),
                             literal.shape().dimensions().end()};
  auto result = TensorOrMemref<T>::Empty(shape, layout);
  assert(literal.size_bytes() == result.buffer->GetByteSize() &&
         "expected buffer sizes to match");
  memcpy(result.buffer->at(0, 0), literal.untyped_data(),
         result.buffer->GetByteSize());
  return result;
}
}  // namespace

absl::StatusOr<InterpreterValue> LiteralToValue(const xla::Literal& literal) {
  if (literal.shape().IsTuple()) {
    auto elements = literal.Clone().DecomposeTuple();
    Tuple result;
    for (auto& element : elements) {
      TF_ASSIGN_OR_RETURN(auto converted, LiteralToValue(element));
      result.values.push_back(
          std::make_shared<InterpreterValue>(std::move(converted)));
    }
    return {{result}};
  }

  if (literal.shape().IsToken()) {
    return absl::UnimplementedError("token arguments are not implemented");
  }

  if (literal.shape().IsArray()) {
    auto type = literal.shape().element_type();
    if (xla::primitive_util::IsF8Type(type)) {
      return absl::UnimplementedError(
          absl::StrCat(xla::primitive_util::LowercasePrimitiveTypeName(type),
                       " not implemented"));
    }
    switch (type) {
      case xla::PRED:
        return {{ArrayLiteralToTensor<bool>(literal)}};
      case xla::S8:
        return {{ArrayLiteralToTensor<int8_t>(literal)}};
      case xla::S16:
        return {{ArrayLiteralToTensor<int16_t>(literal)}};
      case xla::S32:
        return {{ArrayLiteralToTensor<int32_t>(literal)}};
      case xla::S64:
        return {{ArrayLiteralToTensor<int64_t>(literal)}};
      case xla::U8:
        return {{ArrayLiteralToTensor<uint8_t>(literal)}};
      case xla::U16:
        return {{ArrayLiteralToTensor<uint16_t>(literal)}};
      case xla::U32:
        return {{ArrayLiteralToTensor<uint32_t>(literal)}};
      case xla::U64:
        return {{ArrayLiteralToTensor<uint64_t>(literal)}};
      case xla::F16:
        return absl::UnimplementedError("F16 not implemented");
      case xla::F32:
        return {{ArrayLiteralToTensor<float>(literal)}};
      case xla::BF16:
        return absl::UnimplementedError("BF16 not implemented");
      case xla::F64:
        return {{ArrayLiteralToTensor<double>(literal)}};
      case xla::C64:
        return {{ArrayLiteralToTensor<std::complex<float>>(literal)}};
      case xla::C128:
        return {{ArrayLiteralToTensor<std::complex<double>>(literal)}};
      default:
        // Fallthrough intended.
        break;
    }
  }

  return absl::InvalidArgumentError("unexpected literal type");
}

absl::StatusOr<InterpreterValue> LiteralToValue(
    const xla::LiteralProto& literal) {
  TF_ASSIGN_OR_RETURN(auto deserialized,
                      xla::Literal::CreateFromProto(literal));
  return LiteralToValue(deserialized);
}

absl::StatusOr<InterpreterValue> LiteralToValue(
    const xla::LiteralProto& literal, mlir::Type type) {
  TF_ASSIGN_OR_RETURN(auto result, LiteralToValue(literal));
  return {DispatchScalarType(type, [&](auto dummy) -> InterpreterValue {
    TensorOrMemref<decltype(dummy)> cast;
    cast.view = result.View();
    cast.buffer = result.GetBuffer();
    return {cast};
  })};
}

TracedValue ValueToTracedValue(const InterpreterValue& value) {
  TraceInterpreterValueVisitor visitor;
  std::visit(visitor, value.storage);
  return visitor.out;
}

absl::StatusOr<InterpreterValue> TracedValueToValue(
    const TracedValue& traced_value) {
  auto extract = [&](auto dummy, auto& elements) -> InterpreterValue {
    using T = decltype(dummy);
    if (traced_value.is_scalar()) {
      return {static_cast<T>(elements[0])};
    }

    auto result =
        TensorOrMemref<T>::Empty(llvm::to_vector(traced_value.shape()));
    for (auto [index, element] : llvm::zip(result.view.Indices(), elements)) {
      result.at(index) = element;
    }
    return {result};
  };
  auto extract_complex = [&](auto& elements) -> InterpreterValue {
    using T = std::complex<std::decay_t<decltype(elements[0])>>;
    if (traced_value.is_scalar()) {
      return {T{elements[0], elements[1]}};
    }

    auto result =
        TensorOrMemref<T>::Empty(llvm::to_vector(traced_value.shape()));
    int64_t i = 0;
    for (auto it = result.view.Indices().begin(),
              end = result.view.Indices().end();
         it != end; ++it, i += 2) {
      result.at(*it) = {elements[i], elements[i + 1]};
    }
    return {result};
  };
  switch (traced_value.element_type()) {
    case TracedValue::UNKNOWN:
      break;
    case TracedValue::FLOAT:
      if (traced_value.bit_width() == 32) {
        return extract(float{}, traced_value.floats());
      }
      return extract(double{}, traced_value.doubles());
    case TracedValue::UNSIGNED:
      switch (traced_value.bit_width()) {
        case 1:
          return extract(bool{}, traced_value.ints());
        case 8:
          return extract(uint8_t{}, traced_value.uints());
        case 16:
          return extract(uint16_t{}, traced_value.uints());
        case 32:
          return extract(uint32_t{}, traced_value.uints());
        case 64:
          return extract(uint64_t{}, traced_value.uints());
      }
      break;
    case TracedValue::INTEGRAL:
      switch (traced_value.bit_width()) {
        case 8:
          return extract(int8_t{}, traced_value.ints());
        case 16:
          return extract(int16_t{}, traced_value.ints());
        case 32:
          return extract(int32_t{}, traced_value.ints());
        case 64:
          return extract(int64_t{}, traced_value.ints());
      }
      break;
    case TracedValue::COMPLEX:
      switch (traced_value.bit_width()) {
        case 64:
          return extract_complex(traced_value.floats());
        case 128:
          return extract_complex(traced_value.doubles());
      }
      break;
    case TracedValue::TUPLE:
      Tuple result;
      for (const auto& elem : traced_value.tuple_elements()) {
        TF_ASSIGN_OR_RETURN(auto converted, TracedValueToValue(elem));
        result.values.push_back(
            std::make_shared<InterpreterValue>(std::move(converted)));
      }
      return {{std::move(result)}};
  }
  return absl::InvalidArgumentError("unexpected type: " +
                                    traced_value.DebugString());
}

llvm::SmallVector<const InstructionTrace*> FindOpExecutionsInTrace(
    const ExecutionTrace& trace, mlir::Operation* op) {
  llvm::SmallVector<int64_t> region_indices;
  llvm::SmallVector<int64_t> op_indices;

  std::function<void(mlir::Operation*)> get_op_path;
  get_op_path = [&](mlir::Operation* op) {
    auto* parent = op->getParentOp();
    if (!llvm::isa<func::FuncOp>(parent)) {
      get_op_path(parent);
      region_indices.push_back(op->getParentRegion()->getRegionNumber());
    }

    int64_t index = 0;
    while ((op = op->getPrevNode()) != nullptr) ++index;
    op_indices.push_back(index);
  };
  get_op_path(op);

  llvm::SmallVector<const InstructionTrace*> result;
  std::function<void(const RegionTrace& trace, int index)> step;
  step = [&](const RegionTrace& trace, int index) {
    auto& instruction_trace = trace.instructions(op_indices[index]);
    if (region_indices.size() > index) {
      for (const auto& region : instruction_trace.regions()) {
        if (region.region_number() == region_indices[index]) {
          step(region, index + 1);
        }
      }
    } else {
      result.push_back(&instruction_trace);
    }
  };
  step(trace.trace(), 0);

  return result;
}

}  // namespace interpreter
}  // namespace mlir
