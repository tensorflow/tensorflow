/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/ffi/attribute_map.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/ffi/attribute_map.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::ffi {
template <typename OutputVariant, typename... InputTypes>
static OutputVariant Convert(std::variant<InputTypes...> input) {
  return std::visit(
      [](auto&& value) -> OutputVariant {
        return std::forward<decltype(value)>(value);
      },
      std::move(input));
}

// Checks if the given value is in the range of the given integer type.
// Note that only works for integer types where all values can be represented as
// int32_t.
template <typename T>
static bool IsInRange(int32_t value) {
  static_assert(
      std::numeric_limits<T>::is_integer && (sizeof(T) < sizeof(int32_t)),
      "All values of T must be representable as int32_t.");
  return value >= std::numeric_limits<T>::min() &&
         value <= std::numeric_limits<T>::max();
}

static absl::StatusOr<Attribute> ConvertBoolAttr(absl::string_view name,
                                                 mlir::BoolAttr boolean) {
  return static_cast<bool>(boolean.getValue());
}

static absl::StatusOr<Attribute> ConvertStringAttr(absl::string_view name,
                                                   mlir::StringAttr str) {
  return str.getValue().str();
}

static absl::StatusOr<Attribute> ConvertIntegerAttr(absl::string_view name,
                                                    mlir::IntegerAttr integer) {
  if (integer.getType().isUnsignedInteger()) {
    switch (integer.getType().getIntOrFloatBitWidth()) {
      case 8:
        return static_cast<uint8_t>(integer.getUInt());
      case 16:
        return static_cast<uint16_t>(integer.getUInt());
      case 32:
        return static_cast<uint32_t>(integer.getUInt());
      case 64:
        return static_cast<uint64_t>(integer.getUInt());
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported integer attribute bit width for attribute: ", name));
    }
  } else {
    switch (integer.getType().getIntOrFloatBitWidth()) {
      case 8:
        return static_cast<int8_t>(integer.getInt());
      case 16:
        return static_cast<int16_t>(integer.getInt());
      case 32:
        return static_cast<int32_t>(integer.getInt());
      case 64:
        return static_cast<int64_t>(integer.getInt());
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported integer attribute bit width for attribute: ", name));
    }
  }
}

static absl::StatusOr<Attribute> ConvertFloatAttr(absl::string_view name,
                                                  mlir::FloatAttr fp) {
  switch (fp.getType().getIntOrFloatBitWidth()) {
    case 32:
      return static_cast<float>(fp.getValue().convertToFloat());
    case 64:
      return static_cast<double>(fp.getValue().convertToDouble());
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported float attribute bit width for attribute: ", name));
  }
}

static absl::StatusOr<Attribute> ConvertArrayAttr(absl::string_view name,
                                                  mlir::DenseArrayAttr arr) {
  if (auto dense = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  }
  if (auto dense = mlir::dyn_cast<mlir::DenseI16ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  }
  if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  }
  if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  }
  if (auto dense = mlir::dyn_cast<mlir::DenseF32ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  }
  if (auto dense = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported array element type for attribute: ", name));
}

template <typename T>
static std::vector<T> CopyDenseElementsToVec(
    mlir::DenseIntOrFPElementsAttr arr) {
  auto it = arr.getValues<T>();
  return std::vector<T>(it.begin(), it.end());
}

static absl::StatusOr<Attribute> ConvertDenseElementsAttr(
    absl::string_view name, mlir::DenseIntOrFPElementsAttr arr) {
  auto type = arr.getElementType();
  if (type.isInteger()) {
    if (type.isUnsignedInteger()) {
      switch (type.getIntOrFloatBitWidth()) {
        case 8:
          return CopyDenseElementsToVec<uint8_t>(arr);
        case 16:
          return CopyDenseElementsToVec<uint16_t>(arr);
        case 32:
          return CopyDenseElementsToVec<uint32_t>(arr);
        case 64:
          return CopyDenseElementsToVec<uint64_t>(arr);
      }
    } else {
      switch (type.getIntOrFloatBitWidth()) {
        case 8:
          return CopyDenseElementsToVec<int8_t>(arr);
        case 16:
          return CopyDenseElementsToVec<int16_t>(arr);
        case 32:
          return CopyDenseElementsToVec<int32_t>(arr);
        case 64:
          return CopyDenseElementsToVec<int64_t>(arr);
      }
    }
  } else if (type.isIntOrFloat()) {
    switch (type.getIntOrFloatBitWidth()) {
      case 32:
        return CopyDenseElementsToVec<float>(arr);
      case 64:
        return CopyDenseElementsToVec<double>(arr);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported array element type for attribute: ", name));
}

static absl::StatusOr<Attribute> ConvertDictionaryAttr(
    absl::string_view name, mlir::DictionaryAttr dict) {
  TF_ASSIGN_OR_RETURN(auto attrs, BuildAttributesMap(dict));
  return AttributesDictionary{
      std::make_shared<AttributesMap>(std::move(attrs))};
}

absl::StatusOr<AttributesMap> BuildAttributesMap(mlir::DictionaryAttr dict) {
  AttributesMap attributes;
  for (auto& kv : dict) {
    absl::string_view name = kv.getName().strref();
    mlir::Attribute value = kv.getValue();

    // Wraps attribute conversion function into callable object.
    auto convert_with = [&](auto converter_fn) {
      return [&, fn = converter_fn](auto attr) -> absl::Status {
        TF_ASSIGN_OR_RETURN(attributes[name], fn(name, attr));
        return absl::OkStatus();
      };
    };

    TF_RETURN_IF_ERROR(
        llvm::TypeSwitch<mlir::Attribute, absl::Status>(value)
            .Case<mlir::BoolAttr>(convert_with(ConvertBoolAttr))
            .Case<mlir::IntegerAttr>(convert_with(ConvertIntegerAttr))
            .Case<mlir::FloatAttr>(convert_with(ConvertFloatAttr))
            .Case<mlir::DenseArrayAttr>(convert_with(ConvertArrayAttr))
            .Case<mlir::DenseIntOrFPElementsAttr>(
                convert_with(ConvertDenseElementsAttr))
            .Case<mlir::StringAttr>(convert_with(ConvertStringAttr))
            .Case<mlir::DictionaryAttr>(convert_with(ConvertDictionaryAttr))
            .Default([&](mlir::Attribute) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported attribute type for attribute: ", name));
            }));
  }

  return attributes;
}

AttributesMapProto AttributesDictionary::ToProto() const {
  if (attrs != nullptr) {
    return attrs->ToProto();
  }
  return AttributesMapProto();
}

absl::StatusOr<AttributesDictionary> AttributesDictionary::FromProto(
    const AttributesMapProto& proto) {
  TF_ASSIGN_OR_RETURN(auto attrs, AttributesMap::FromProto(proto));
  return AttributesDictionary{std::make_shared<AttributesMap>(attrs)};
}

AttributesMapProto AttributesMap::ToProto() const {
  AttributesMapProto proto;
  for (const auto& [key, value] : *this) {
    (*proto.mutable_attrs())[key] = value.ToProto();
  }
  return proto;
}

absl::StatusOr<AttributesMap> AttributesMap::FromProto(
    const AttributesMapProto& proto) {
  AttributesMap result;
  for (const auto& [key, value] : proto.attrs()) {
    TF_ASSIGN_OR_RETURN(result[key], Attribute::FromProto(value));
  }
  return result;
}

absl::StatusOr<Attribute> Attribute::FromProto(const AttributeProto& proto) {
  Attribute attribute;
  switch (proto.value_case()) {
    case AttributeProto::kScalar:
      return Scalar::FromProto(proto.scalar());
    case AttributeProto::kArray:
      return Array::FromProto(proto.array());
    case AttributeProto::kStr:
      return Attribute(proto.str());
    case AttributeProto::kDict:
      return AttributesDictionary::FromProto(proto.dict());
    default:
      return absl::InvalidArgumentError("Unsupported attribute type");
  }
}

xla::ffi::AttributeProto Attribute::ToProto() const {
  AttributeProto proto;
  std::visit(
      [&](auto&& value) {
        using U = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<U, Scalar>) {
          *proto.mutable_scalar() = value.ToProto();
        } else if constexpr (std::is_same_v<U, Array>) {
          *proto.mutable_array() = value.ToProto();
        } else if constexpr (std::is_same_v<U, std::string>) {
          proto.set_str(value);
        } else if constexpr (std::is_same_v<U, AttributesDictionary>) {
          *proto.mutable_dict() = value.ToProto();
        } else {
          static_assert(false, "Unsupported attribute type");
        }
      },
      AsVariant());
  return proto;
}

ScalarProto Scalar::ToProto() const {
  ScalarProto proto;
  std::visit(
      [&](auto&& value) {
        using U = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<U, bool>) {
          proto.set_b(value);
        } else if constexpr (std::is_same_v<U, int8_t>) {
          proto.set_i8(value);
        } else if constexpr (std::is_same_v<U, int16_t>) {
          proto.set_i16(value);
        } else if constexpr (std::is_same_v<U, int32_t>) {
          proto.set_i32(value);
        } else if constexpr (std::is_same_v<U, int64_t>) {
          proto.set_i64(value);
        } else if constexpr (std::is_same_v<U, uint8_t>) {
          proto.set_u8(value);
        } else if constexpr (std::is_same_v<U, uint16_t>) {
          proto.set_u16(value);
        } else if constexpr (std::is_same_v<U, uint32_t>) {
          proto.set_u32(value);
        } else if constexpr (std::is_same_v<U, uint64_t>) {
          proto.set_u64(value);
        } else if constexpr (std::is_same_v<U, float>) {
          proto.set_f32(value);
        } else if constexpr (std::is_same_v<U, double>) {
          proto.set_f64(value);
        } else {
          static_assert(false, "Unsupported scalar type");
        }
      },
      AsVariant());
  return proto;
}

absl::StatusOr<Scalar> Scalar::FromProto(const ScalarProto& proto) {
  switch (proto.value_case()) {
    case ScalarProto::kB:
      return proto.b();
    case ScalarProto::kI8:
      if (!IsInRange<int8_t>(proto.i8())) {
        return absl::InvalidArgumentError(
            "Integer value out of range for int8_t");
      }
      return static_cast<int8_t>(proto.i8());
    case ScalarProto::kI16:
      if (!IsInRange<int16_t>(proto.i16())) {
        return absl::InvalidArgumentError(
            "Integer value out of range for int16_t");
      }
      return static_cast<int16_t>(proto.i16());
    case ScalarProto::kI32:
      return proto.i32();
    case ScalarProto::kI64:
      return proto.i64();
    case ScalarProto::kU8:
      if (!IsInRange<uint8_t>(proto.u8())) {
        return absl::InvalidArgumentError(
            "Integer value out of range for uint8_t");
      }
      return static_cast<uint8_t>(proto.u8());
    case ScalarProto::kU16:
      if (!IsInRange<uint16_t>(proto.u16())) {
        return absl::InvalidArgumentError(
            "Integer value out of range for uint16_t");
      }
      return static_cast<uint16_t>(proto.u16());
    case ScalarProto::kU32:
      return proto.u32();
    case ScalarProto::kU64:
      return proto.u64();
    case ScalarProto::kF32:
      return proto.f32();
    case ScalarProto::kF64:
      return proto.f64();
    default:
      return absl::InvalidArgumentError("Unsupported scalar type");
  }
}

ArrayProto Array::ToProto() const {
  ArrayProto proto;
  std::visit(
      [&](auto&& value) {
        using U = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<U, std::vector<int8_t>>) {
          proto.mutable_i8()->mutable_values()->Assign(value.begin(),
                                                       value.end());
        } else if constexpr (std::is_same_v<U, std::vector<int16_t>>) {
          proto.mutable_i16()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<int32_t>>) {
          proto.mutable_i32()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<int64_t>>) {
          proto.mutable_i64()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<uint8_t>>) {
          proto.mutable_u8()->mutable_values()->Assign(value.begin(),
                                                       value.end());
        } else if constexpr (std::is_same_v<U, std::vector<uint16_t>>) {
          proto.mutable_u16()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<uint32_t>>) {
          proto.mutable_u32()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<uint64_t>>) {
          proto.mutable_u64()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<float>>) {
          proto.mutable_f32()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else if constexpr (std::is_same_v<U, std::vector<double>>) {
          proto.mutable_f64()->mutable_values()->Assign(value.begin(),
                                                        value.end());
        } else {
          static_assert(false, "Unsupported array type");
        }
      },
      AsVariant());
  return proto;
}

absl::StatusOr<Array> Array::FromProto(const ArrayProto& proto) {
  switch (proto.value_case()) {
    case ArrayProto::kI8:
      if (!absl::c_all_of(proto.i8().values(), IsInRange<int8_t>)) {
        return absl::InvalidArgumentError(
            "Integer value out of range for int8_t");
      }
      return std::vector<int8_t>(proto.i8().values().begin(),
                                 proto.i8().values().end());
    case ArrayProto::kI16:
      if (!absl::c_all_of(proto.i16().values(), IsInRange<int16_t>)) {
        return absl::InvalidArgumentError(
            "Integer value out of range for int16_t");
      }
      return std::vector<int16_t>(proto.i16().values().begin(),
                                  proto.i16().values().end());
    case ArrayProto::kI32:
      return std::vector<int32_t>(proto.i32().values().begin(),
                                  proto.i32().values().end());
    case ArrayProto::kI64:
      return std::vector<int64_t>(proto.i64().values().begin(),
                                  proto.i64().values().end());
    case ArrayProto::kU8:
      if (!absl::c_all_of(proto.u8().values(), IsInRange<uint8_t>)) {
        return absl::InvalidArgumentError(
            "Integer value out of range for uint8_t");
      }
      return std::vector<uint8_t>(proto.u8().values().begin(),
                                  proto.u8().values().end());
    case ArrayProto::kU16:
      if (!absl::c_all_of(proto.u16().values(), IsInRange<uint16_t>)) {
        return absl::InvalidArgumentError(
            "Integer value out of range for uint16_t");
      }
      return std::vector<uint16_t>(proto.u16().values().begin(),
                                   proto.u16().values().end());
    case ArrayProto::kU32:
      return std::vector<uint32_t>(proto.u32().values().begin(),
                                   proto.u32().values().end());
    case ArrayProto::kU64:
      return std::vector<uint64_t>(proto.u64().values().begin(),
                                   proto.u64().values().end());
    case ArrayProto::kF32:
      return std::vector<float>(proto.f32().values().begin(),
                                proto.f32().values().end());
    case ArrayProto::kF64:
      return std::vector<double>(proto.f64().values().begin(),
                                 proto.f64().values().end());
    default:
      return absl::InvalidArgumentError("Unsupported array type");
  }
}

FlatAttributeProto FlatAttribute::ToProto() const {
  FlatAttributeProto proto;
  std::visit(
      [&](auto&& value) {
        using U = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<U, Scalar>) {
          *proto.mutable_scalar() = value.ToProto();
        } else if constexpr (std::is_same_v<U, Array>) {
          *proto.mutable_array() = value.ToProto();
        } else if constexpr (std::is_same_v<U, std::string>) {
          proto.set_str(value);
        } else {
          static_assert(false, "Unsupported flat attribute type");
        }
      },
      AsVariant());
  return proto;
}

absl::StatusOr<FlatAttribute> FlatAttribute::FromProto(
    const FlatAttributeProto& proto) {
  switch (proto.value_case()) {
    case FlatAttributeProto::kScalar:
      return Scalar::FromProto(proto.scalar());
    case FlatAttributeProto::kArray:
      return Array::FromProto(proto.array());
    case FlatAttributeProto::kStr:
      return proto.str();
    default:
      return absl::InvalidArgumentError("Unsupported flat attribute type");
  }
}

FlatAttributesMapProto FlatAttributesMap::ToProto() const {
  FlatAttributesMapProto proto;
  for (const auto& [key, value] : *this) {
    (*proto.mutable_attrs())[key] = value.ToProto();
  }
  return proto;
}

absl::StatusOr<FlatAttributesMap> FlatAttributesMap::FromProto(
    const FlatAttributesMapProto& proto) {
  FlatAttributesMap result;
  for (const auto& [key, value] : proto.attrs()) {
    TF_ASSIGN_OR_RETURN(result[key], FlatAttribute::FromProto(value));
  }
  return result;
}

Attribute::Attribute(const FlatAttribute& flat)
    : Attribute(Convert<Attribute>(flat)) {}

bool operator==(const AttributesDictionary& lhs,
                const AttributesDictionary& rhs) {
  if (lhs.attrs == nullptr) {
    return rhs.attrs == nullptr;
  }
  if (rhs.attrs == nullptr) {
    return false;
  }
  return *lhs.attrs == *rhs.attrs;
}

}  // namespace xla::ffi
