/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/python/ifrt/ir/ifrt_dialect.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

// Generated definitions.
#include "xla/python/ifrt/ir/ifrt_dialect.cc.inc"
#include "xla/python/ifrt/ir/sharding_param.h"
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/ifrt_types.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/ifrt_attrs.cc.inc"

namespace xla {
namespace ifrt {

class IfrtAsmDialectInterface : public mlir::OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute attr,
                       llvm::raw_ostream& os) const override;
};

void IfrtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/python/ifrt/ir/ifrt_types.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/python/ifrt/ir/ifrt_attrs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/ifrt_ops.cc.inc"
      >();
  addInterfaces<IfrtAsmDialectInterface>();
}

IfrtAsmDialectInterface::AliasResult IfrtAsmDialectInterface::getAlias(
    mlir::Attribute attr, llvm::raw_ostream& os) const {
  if (auto devices = llvm::dyn_cast<IfrtDevicesAttr>(attr);
      devices != nullptr && devices.getIds().size() > 4) {
    os << "devices";
    return AliasResult::FinalAlias;
  }
  return AliasResult::NoAlias;
}

mlir::LogicalResult IfrtDialect::verifyOperationAttribute(
    mlir::Operation* op, mlir::NamedAttribute attr) {
  if (attr.getName() == kIfrtFunctionAttrName) {
    if (!llvm::isa<mlir::func::FuncOp>(op)) {
      return op->emitOpError() << "has `" << kIfrtFunctionAttrName
                               << "` attr but is not a function";
    }
    if (!attr.getValue().isa<mlir::UnitAttr>()) {
      return op->emitOpError() << "has `" << kIfrtFunctionAttrName
                               << "` attr that is not a UnitAttr";
    }
  }
  return mlir::success();
}

mlir::LogicalResult IfrtDialect::verifyRegionArgAttribute(
    mlir::Operation* op, unsigned regionIndex, unsigned argIndex,
    mlir::NamedAttribute attr) {
  if (attr.getName() == kIfrtDonatedArgAttrName) {
    if (!llvm::isa<mlir::func::FuncOp>(op)) {
      return op->emitOpError() << "has `" << kIfrtDonatedArgAttrName
                               << "` arg attr but is not a function";
    }
    if (!attr.getValue().isa<mlir::UnitAttr>()) {
      return op->emitOpError() << "has `" << kIfrtDonatedArgAttrName
                               << "` arg attr that is not a UnitAttr";
    }
    if (!op->hasAttr(kIfrtFunctionAttrName)) {
      return op->emitOpError()
             << "has `" << kIfrtDonatedArgAttrName << "` arg attr but not has `"
             << kIfrtFunctionAttrName << "` attr";
    }
  }
  return mlir::success();
}

mlir::LogicalResult IfrtShardingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ShardingParam sharding) {
  return sharding.verify(emitError);
}

llvm::ArrayRef<int> IfrtArrayType::getDevices() const {
  return getDevicesAttr().getIds();
}

mlir::LogicalResult IfrtArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, ShardingParam sharding,
    IfrtDevicesAttr devices) {
  if (mlir::failed(sharding.verify(emitError))) {
    return mlir::failure();
  }

  if (shape.getRank() != sharding.dim_shards().size()) {
    return emitError() << "Requires dim shards to have the same rank as the "
                          "array. Array rank is "
                       << shape.getRank() << " vs dim shards rank of "
                       << sharding.dim_shards().size();
  }

  int devices_in_mesh = 1;
  for (const int axis_size : sharding.minor_to_major().axis_sizes) {
    devices_in_mesh *= axis_size;
  }
  if (llvm::ArrayRef<int> ids = devices.getIds();
      devices_in_mesh != ids.size()) {
    return emitError() << "Requires the same amount of `devices` and from "
                          "`sharding`. Actual: "
                       << ids.size() << " vs " << devices_in_mesh;
  }

  return mlir::success();
}

IfrtDevicesAttr::operator llvm::ArrayRef<int>() const { return getIds(); }

mlir::LogicalResult IfrtDevicesAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int> ids) {
  llvm::SmallSet<int, 4> device_set;
  for (int id : ids) {
    if (id < 0) {
      return emitError() << "Device list has negative id " << id;
    }
    if (auto [unused_it, inserted] = device_set.insert(id); !inserted) {
      return emitError() << "Device list has duplicate id " << id;
    }
  }

  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla
