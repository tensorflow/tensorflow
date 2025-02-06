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

#include "xla/python/ifrt/ir/vifrt_bytecode.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"

// Tag to be passed to `-debug-only` argument so that the mlir opt tool prints
// the debug info for this bytecode writer.
#define DEBUG_TYPE "vifrt-bytecode"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Util macros copied from VHLO and MHLO bytecode writers.
#define _EXTRACT_AFTER(a, b) \
  llvm::StringRef(a).substr(llvm::StringRef(a).find(b))

#define _LOG_CALL_TO(func)                                              \
  LLVM_DEBUG(llvm::errs() << "Called: "                                 \
                          << _EXTRACT_AFTER(LLVM_PRETTY_FUNCTION, func) \
                          << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED(typeOrAttr)                                  \
  LLVM_DEBUG(llvm::errs() << "***Not Implemented: " << typeOrAttr << " " \
                          << LLVM_PRETTY_FUNCTION << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace vifrt_encoding {

// This enum contains markers used to indicate which attribute is being decoded.
// The order of the enum values must not be changed because changes will break
// compatibility with older bytecode.
//
// When adding a new attribute to the dialect, update the code wherever
// "ADD ATTRIBUTE" is present in this file to ensure stable bytecode.
enum AttributeMarker {
  kDevicesV1Attr = 0,
  kUnspecifiedShardingV1Attr = 1,
  kShardingParamV1Attr = 2,
  kIntervalV1Attr = 3,
  kMappingV1Attr = 4,
  kArrayMappingV1Attr = 5,
  kTypeV1Attr = 6,
  // Always increment the enum value; Next available code: 7
  // ADD ATTRIBUTE: Add an enum value for new VIFRT attr.
};

// This enum contains markers used to indicate which type is being decoded.
// The order of the enum values must not be changed because changes will break
// compatibility with older bytecode.
//
// When adding a new type to the dialect, update the code wherever "ADD TYPE"
// is present in this file to ensure stable bytecode.
enum TypeMarker {
  kArrayV1Type = 0,
  kControlV1Type = 1,
  kFunctionV1Type = 2,
  // Always increment the enum value; Next available code: 3
  // ADD TYPE: Add an enum value for new type.
};

}  // namespace vifrt_encoding
}  // namespace

//===----------------------------------------------------------------------===//
// VifrtBytecodeInterface
//===----------------------------------------------------------------------===//

namespace xla {
namespace ifrt {

namespace {

// Implements the BytecodeDialectInterface for the VIFRT dialect.
class VifrtBytecodeInterface : public mlir::BytecodeDialectInterface {
 public:
  explicit VifrtBytecodeInterface(mlir::Dialect *dialect)
      : mlir::BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // Methods invoked by superclass when an attr from VIFRT dialect is found.
  mlir::Attribute readAttribute(
      mlir::DialectBytecodeReader &reader) const override;
  mlir::LogicalResult writeAttribute(
      mlir::Attribute attr, mlir::DialectBytecodeWriter &writer) const override;

  // ADD ATTRIBUTE: Include a read method for each attribute in VIFRT.
  VifrtDevicesV1Attr readDevicesV1Attr(
      mlir::DialectBytecodeReader &reader) const;
  VifrtUnspecifiedShardingV1Attr readUnspecifiedShardingV1Attr(
      mlir::DialectBytecodeReader &reader) const;
  VifrtShardingParamV1Attr readShardingParamV1Attr(
      mlir::DialectBytecodeReader &reader) const;
  VifrtIntervalV1Attr readIntervalV1Attr(
      mlir::DialectBytecodeReader &reader) const;
  VifrtMappingV1Attr readMappingV1Attr(
      mlir::DialectBytecodeReader &reader) const;
  VifrtArrayMappingV1Attr readArrayMappingV1Attr(
      mlir::DialectBytecodeReader &reader) const;
  VifrtTypeV1Attr readTypeV1Attr(mlir::DialectBytecodeReader &reader) const;

  // ADD ATTRIBUTE: Include a write method for each attribute in VIFRT
  void write(VifrtDevicesV1Attr attr,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtUnspecifiedShardingV1Attr attr,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtShardingParamV1Attr attr,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtIntervalV1Attr attr,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtMappingV1Attr attr,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtArrayMappingV1Attr attr,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtTypeV1Attr attr, mlir::DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  // Methods invoked by superclass when a type from VIFRT dialect is found.
  mlir::Type readType(mlir::DialectBytecodeReader &reader) const override;
  mlir::LogicalResult writeType(
      mlir::Type type, mlir::DialectBytecodeWriter &writer) const override;

  // ADD TYPE: Include a read method for each type in VIFRT
  VifrtArrayV1Type readArrayV1Type(mlir::DialectBytecodeReader &reader) const;
  VifrtControlV1Type readControlV1Type(
      mlir::DialectBytecodeReader &reader) const;
  VifrtFunctionV1Type readFunctionV1Type(
      mlir::DialectBytecodeReader &reader) const;

  // ADD TYPE: Include a write method for each type in VIFRT
  void write(VifrtArrayV1Type type, mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtControlV1Type type,
             mlir::DialectBytecodeWriter &writer) const;
  void write(VifrtFunctionV1Type type,
             mlir::DialectBytecodeWriter &writer) const;
};

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// ADD ATTRIBUTE: Update the switch to include a branch for the new attr.
mlir::Attribute VifrtBytecodeInterface::readAttribute(
    mlir::DialectBytecodeReader &reader) const {
  uint64_t marker;
  if (mlir::failed(reader.readVarInt(marker))) {
    reader.emitError() << "Failed to read attribute marker";
    return mlir::Attribute();
  }
  switch (marker) {
    case vifrt_encoding::kDevicesV1Attr:
      return readDevicesV1Attr(reader);
    case vifrt_encoding::kUnspecifiedShardingV1Attr:
      return readUnspecifiedShardingV1Attr(reader);
    case vifrt_encoding::kShardingParamV1Attr:
      return readShardingParamV1Attr(reader);
    case vifrt_encoding::kIntervalV1Attr:
      return readIntervalV1Attr(reader);
    case vifrt_encoding::kMappingV1Attr:
      return readMappingV1Attr(reader);
    case vifrt_encoding::kArrayMappingV1Attr:
      return readArrayMappingV1Attr(reader);
    case vifrt_encoding::kTypeV1Attr:
      return readTypeV1Attr(reader);
    default:
      reader.emitError() << "unknown VIFRT attribute marker: " << marker;
      return mlir::Attribute();
  }
}

// ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
mlir::LogicalResult VifrtBytecodeInterface::writeAttribute(
    mlir::Attribute attr, mlir::DialectBytecodeWriter &writer) const {
  return llvm::TypeSwitch<mlir::Attribute, mlir::LogicalResult>(attr)
      .Case<VifrtDevicesV1Attr, VifrtUnspecifiedShardingV1Attr,
            VifrtShardingParamV1Attr, VifrtIntervalV1Attr, VifrtMappingV1Attr,
            VifrtArrayMappingV1Attr, VifrtTypeV1Attr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return mlir::success();
      })
      .Default([&](mlir::Attribute attr) {
        LOG_NOT_IMPLEMENTED(attr);
        return mlir::failure();
      });
}

VifrtDevicesV1Attr VifrtBytecodeInterface::readDevicesV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> device_ids;
  if (mlir::failed(reader.readSignedVarInts(device_ids))) {
    reader.emitError() << "Failed to read VifrtDevicesV1Attr";
    return VifrtDevicesV1Attr();
  }
  return VifrtDevicesV1Attr::get(
      getContext(),
      llvm::SmallVector<int>(device_ids.begin(), device_ids.end()));
}

void VifrtBytecodeInterface::write(VifrtDevicesV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kDevicesV1Attr);
  writer.writeList(attr.getIds(), [&](int device_id) {
    return writer.writeSignedVarInt(static_cast<int64_t>(device_id));
  });
}

VifrtUnspecifiedShardingV1Attr
VifrtBytecodeInterface::readUnspecifiedShardingV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return VifrtUnspecifiedShardingV1Attr::get(getContext());
}

void VifrtBytecodeInterface::write(VifrtUnspecifiedShardingV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kUnspecifiedShardingV1Attr);
}

VifrtShardingParamV1Attr VifrtBytecodeInterface::readShardingParamV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> dim_shards;
  llvm::SmallVector<int64_t> permutation;
  llvm::SmallVector<int64_t> axis_sizes;
  if (mlir::failed(reader.readSignedVarInts(dim_shards)) ||
      mlir::failed(reader.readSignedVarInts(permutation)) ||
      mlir::failed(reader.readSignedVarInts(axis_sizes))) {
    reader.emitError() << "Failed to read VifrtShardingParamV1Attr";
    return VifrtShardingParamV1Attr();
  }
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation =
      llvm::SmallVector<int, 4>(permutation.begin(), permutation.end());
  minor_to_major.axis_sizes =
      llvm::SmallVector<int, 4>(axis_sizes.begin(), axis_sizes.end());
  ShardingParam sharding_param(
      std::vector(dim_shards.begin(), dim_shards.end()),
      std::move(minor_to_major));
  return VifrtShardingParamV1Attr::get(getContext(), std::move(sharding_param));
}

void VifrtBytecodeInterface::write(VifrtShardingParamV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kShardingParamV1Attr);
  auto sharding = attr.getSharding();
  writer.writeSignedVarInts(sharding.dim_shards());
  writer.writeList(sharding.minor_to_major().permutation, [&](int value) {
    return writer.writeSignedVarInt(static_cast<int64_t>(value));
  });
  writer.writeList(sharding.minor_to_major().axis_sizes, [&](int value) {
    return writer.writeSignedVarInt(static_cast<int64_t>(value));
  });
}

VifrtIntervalV1Attr VifrtBytecodeInterface::readIntervalV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t start;
  int64_t end;
  int64_t step;
  if (mlir::failed(reader.readSignedVarInt(start)) ||
      mlir::failed(reader.readSignedVarInt(end)) ||
      mlir::failed(reader.readSignedVarInt(step))) {
    reader.emitError() << "Failed to read VifrtIntervalV1Attr";
    return VifrtIntervalV1Attr();
  }
  return VifrtIntervalV1Attr::get(getContext(), static_cast<int>(start),
                                  static_cast<int>(end),
                                  static_cast<int>(step));
}

void VifrtBytecodeInterface::write(VifrtIntervalV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kIntervalV1Attr);
  writer.writeSignedVarInt(static_cast<int64_t>(attr.getStart()));
  writer.writeSignedVarInt(static_cast<int64_t>(attr.getEnd()));
  writer.writeSignedVarInt(static_cast<int64_t>(attr.getStep()));
}

VifrtMappingV1Attr VifrtBytecodeInterface::readMappingV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  VifrtIntervalV1Attr from_shards;
  VifrtIntervalV1Attr to_shards;
  if (mlir::failed(reader.readAttribute<VifrtIntervalV1Attr>(from_shards)) ||
      mlir::failed(reader.readAttribute<VifrtIntervalV1Attr>(to_shards))) {
    reader.emitError() << "Failed to read VifrtMappingV1Attr";
    return VifrtMappingV1Attr();
  }
  return VifrtMappingV1Attr::get(getContext(), from_shards, to_shards);
}

void VifrtBytecodeInterface::write(VifrtMappingV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kMappingV1Attr);
  writer.writeAttribute(attr.getFromShards());
  writer.writeAttribute(attr.getToShards());
}

VifrtArrayMappingV1Attr VifrtBytecodeInterface::readArrayMappingV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  int64_t in_array_index;
  int64_t out_array_index;
  llvm::SmallVector<VifrtMappingV1Attr> mappings;
  if (mlir::failed(reader.readSignedVarInt(in_array_index)) ||
      mlir::failed(reader.readSignedVarInt(out_array_index)) ||
      mlir::failed(reader.readAttributes(mappings))) {
    reader.emitError() << "Failed to read VifrtArrayMappingV1Attr";
    return VifrtArrayMappingV1Attr();
  }
  mlir::ArrayAttr array_attr = mlir::ArrayAttr::get(
      getContext(),
      llvm::SmallVector<mlir::Attribute>(mappings.begin(), mappings.end()));
  return VifrtArrayMappingV1Attr::get(
      getContext(), static_cast<int32_t>(in_array_index),
      static_cast<int32_t>(out_array_index), array_attr);
}

void VifrtBytecodeInterface::write(VifrtArrayMappingV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kArrayMappingV1Attr);
  writer.writeSignedVarInt(attr.getInArrayIndex());
  writer.writeSignedVarInt(attr.getOutArrayIndex());
  writer.writeAttributes(attr.getMappings().getValue());
}

VifrtTypeV1Attr VifrtBytecodeInterface::readTypeV1Attr(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  mlir::Type type;
  if (mlir::failed(reader.readType(type))) {
    reader.emitError() << "Failed to read VifrtTypeV1Attr";
    return VifrtTypeV1Attr();
  };
  return VifrtTypeV1Attr::get(getContext(), type);
}

void VifrtBytecodeInterface::write(VifrtTypeV1Attr attr,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kTypeV1Attr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// ADD TYPE: Update the case selection to include the new type.
mlir::Type VifrtBytecodeInterface::readType(
    mlir::DialectBytecodeReader &reader) const {
  uint64_t marker;
  if (mlir::failed(reader.readVarInt(marker))) {
    reader.emitError() << "Failed to read type marker";
    return mlir::Type();
  }
  switch (marker) {
    case vifrt_encoding::kArrayV1Type:
      return readArrayV1Type(reader);
    case vifrt_encoding::kControlV1Type:
      return readControlV1Type(reader);
    case vifrt_encoding::kFunctionV1Type:
      return readFunctionV1Type(reader);
    default:
      reader.emitError() << "unknown VIFRT type marker: " << marker;
      return mlir::Type();
  }
}

// ADD TYPE: Update the case selection to include the new type.
// If this method returns failure, the string serialization is used in the
// bytecode.
mlir::LogicalResult VifrtBytecodeInterface::writeType(
    mlir::Type type, mlir::DialectBytecodeWriter &writer) const {
  return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
      .Case<VifrtArrayV1Type, VifrtControlV1Type, VifrtFunctionV1Type>(
          [&](auto type) {
            LOG_WRITE_CALL;
            return write(type, writer), mlir::success();
          })
      .Default([&](mlir::Type type) {
        LOG_NOT_IMPLEMENTED(type);
        return mlir::failure();
      });
}

VifrtArrayV1Type VifrtBytecodeInterface::readArrayV1Type(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  mlir::RankedTensorType shape;
  mlir::Attribute sharding;
  VifrtDevicesV1Attr devices;
  mlir::StringAttr memory_kind;
  mlir::StringAttr layout;
  if (mlir::failed(reader.readType<mlir::RankedTensorType>(shape)) ||
      mlir::failed(reader.readAttribute(sharding)) ||
      mlir::failed(reader.readAttribute<VifrtDevicesV1Attr>(devices)) ||
      mlir::failed(reader.readAttribute<mlir::StringAttr>(memory_kind)) ||
      mlir::failed(reader.readAttribute<mlir::StringAttr>(layout))) {
    reader.emitError() << "Failed to read VifrtArrayV1Type";
    return VifrtArrayV1Type();
  }
  return VifrtArrayV1Type::get(getContext(), shape, sharding, devices,
                               memory_kind, layout);
}

void VifrtBytecodeInterface::write(VifrtArrayV1Type type,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kArrayV1Type);
  writer.writeType(type.getShape());
  writer.writeAttribute(type.getShardingAttr());
  writer.writeAttribute(type.getDevicesAttr());
  writer.writeAttribute(type.getMemoryKindAttr());
  writer.writeAttribute(type.getLayoutAttr());
}

VifrtControlV1Type VifrtBytecodeInterface::readControlV1Type(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return VifrtControlV1Type::get(getContext());
}

void VifrtBytecodeInterface::write(VifrtControlV1Type type,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kControlV1Type);
}

VifrtFunctionV1Type VifrtBytecodeInterface::readFunctionV1Type(
    mlir::DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<mlir::Type> inputs;
  llvm::SmallVector<mlir::Type> outputs;
  if (mlir::failed(reader.readTypes(inputs)) ||
      mlir::failed(reader.readTypes(outputs))) {
    reader.emitError() << "Failed to read VifrtFunctionV1Type";
    return VifrtFunctionV1Type();
  }
  return VifrtFunctionV1Type::get(getContext(), inputs, outputs);
}

void VifrtBytecodeInterface::write(VifrtFunctionV1Type type,
                                   mlir::DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vifrt_encoding::kFunctionV1Type);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getOutputs());
}

}  // namespace

void addBytecodeInterface(VifrtDialect *dialect) {
  dialect->addInterfaces<VifrtBytecodeInterface>();
}

}  // namespace ifrt
}  // namespace xla
