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
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/ffi/call_frame.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::ffi {

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertBoolAttr(
    absl::string_view name, mlir::BoolAttr boolean) {
  return static_cast<bool>(boolean.getValue());
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertStringAttr(
    absl::string_view name, mlir::StringAttr str) {
  return str.getValue().str();
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertIntegerAttr(
    absl::string_view name, mlir::IntegerAttr integer) {
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

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertFloatAttr(
    absl::string_view name, mlir::FloatAttr fp) {
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

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertArrayAttr(
    absl::string_view name, mlir::DenseArrayAttr arr) {
  if (auto dense = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseI16ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseF32ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported array element type for attribute: ", name));
  }
}

template <typename T>
static std::vector<T> CopyDenseElementsToVec(
    mlir::DenseIntOrFPElementsAttr arr) {
  auto it = arr.getValues<T>();
  return std::vector<T>(it.begin(), it.end());
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertDenseElementsAttr(
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

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertDictionaryAttr(
    absl::string_view name, mlir::DictionaryAttr dict) {
  TF_ASSIGN_OR_RETURN(auto attrs, BuildAttributesMap(dict));
  return CallFrameBuilder::Dictionary{
      std::make_shared<CallFrameBuilder::AttributesMap>(std::move(attrs))};
}

absl::StatusOr<CallFrameBuilder::AttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict) {
  CallFrameBuilder::AttributesMap attributes;
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

}  // namespace xla::ffi
