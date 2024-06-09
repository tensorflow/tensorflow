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
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/ffi/call_frame.h"
#include "tsl/platform/errors.h"

using FlatAttribute = xla::ffi::CallFrameBuilder::FlatAttribute;
using FlatAttributesMap = xla::ffi::CallFrameBuilder::FlatAttributesMap;

namespace xla::ffi {

absl::StatusOr<FlatAttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict) {
  FlatAttributesMap attributes;
  for (auto& kv : dict) {
    std::string_view name = kv.getName().strref();

    auto boolean = [&](mlir::BoolAttr boolean) {
      attributes[name] = static_cast<bool>(boolean.getValue());
      return absl::OkStatus();
    };

    auto integer = [&](mlir::IntegerAttr integer) {
      if (integer.getType().isUnsignedInteger()) {
        switch (integer.getType().getIntOrFloatBitWidth()) {
          case 8:
            attributes[name] = static_cast<uint8_t>(integer.getUInt());
            return absl::OkStatus();
          case 16:
            attributes[name] = static_cast<uint16_t>(integer.getUInt());
            return absl::OkStatus();
          case 32:
            attributes[name] = static_cast<uint32_t>(integer.getUInt());
            return absl::OkStatus();
          case 64:
            attributes[name] = static_cast<uint64_t>(integer.getUInt());
            return absl::OkStatus();
          default:
            return absl::InvalidArgumentError(absl::StrCat(
                "Unsupported integer attribute bit width for attribute: ",
                name));
        }
      } else {
        switch (integer.getType().getIntOrFloatBitWidth()) {
          case 8:
            attributes[name] = static_cast<int8_t>(integer.getInt());
            return absl::OkStatus();
          case 16:
            attributes[name] = static_cast<int16_t>(integer.getInt());
            return absl::OkStatus();
          case 32:
            attributes[name] = static_cast<int32_t>(integer.getInt());
            return absl::OkStatus();
          case 64:
            attributes[name] = static_cast<int64_t>(integer.getInt());
            return absl::OkStatus();
          default:
            return absl::InvalidArgumentError(absl::StrCat(
                "Unsupported integer attribute bit width for attribute: ",
                name));
        }
      }
    };

    auto fp = [&](mlir::FloatAttr fp) {
      switch (fp.getType().getIntOrFloatBitWidth()) {
        case 32:
          attributes[name] = static_cast<float>(fp.getValue().convertToFloat());
          return absl::OkStatus();
        case 64:
          attributes[name] =
              static_cast<double>(fp.getValue().convertToDouble());
          return absl::OkStatus();
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported float attribute bit width for attribute: ", name));
      }
    };

    auto arr = [&](mlir::DenseArrayAttr arr) {
      if (auto dense = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseI16ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseF32ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else if (auto dense = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(arr)) {
        attributes[name] = dense.asArrayRef().vec();
        return absl::OkStatus();
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported array element type for attribute: ", name));
      }
    };

    auto str = [&](mlir::StringAttr str) {
      attributes[name] = str.getValue().str();
      return absl::OkStatus();
    };

    TF_RETURN_IF_ERROR(
        llvm::TypeSwitch<mlir::Attribute, absl::Status>(kv.getValue())
            .Case<mlir::BoolAttr>(boolean)
            .Case<mlir::IntegerAttr>(integer)
            .Case<mlir::FloatAttr>(fp)
            .Case<mlir::DenseArrayAttr>(arr)
            .Case<mlir::StringAttr>(str)
            .Default([&](mlir::Attribute) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported attribute type for attribute: ", name));
            }));
  }
  return attributes;
}
}  // namespace xla::ffi
