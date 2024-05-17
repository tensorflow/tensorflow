/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"

// Generated definitions.
#include "xla/python/ifrt/ir/ifrt_dialect.cc.inc"
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
  } else if (auto mapping = llvm::dyn_cast<IfrtArrayMappingAttr>(attr);
             mapping != nullptr && mapping.getMappings().size() > 2) {
    os << "array_mapping";
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
    if (!mlir::isa<mlir::UnitAttr>(attr.getValue())) {
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
    if (!mlir::isa<mlir::UnitAttr>(attr.getValue())) {
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

//===----------------------------------------------------------------------===//
// IfrtShardingParamAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtShardingParamAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ShardingParam sharding_param, mlir::StringAttr memory_kind) {
  return sharding_param.verify(emitError);
}

mlir::LogicalResult IfrtShardingParamAttr::CanApplyTo(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, llvm::ArrayRef<int> device_ids) const {
  return getSharding().CanApplyTo(emitError, shape, device_ids);
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtShardingParamAttr::GlobalShapeFromLocalShape(
    llvm::ArrayRef<int64_t> local_shape) const {
  return getSharding().GlobalShapeFromLocalShape(local_shape);
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtShardingParamAttr::LocalShapeFromGlobalShape(
    llvm::ArrayRef<int64_t> global_shape) const {
  return getSharding().LocalShapeFromGlobalShape(global_shape);
}

// Returns the number of devices the sharding applies to.
int IfrtShardingParamAttr::NumDevices() const {
  return getSharding().NumDevices();
};

xla::ifrt::MemoryKind IfrtShardingParamAttr::MemoryKind() const {
  return getMemoryKind() == nullptr
             ? xla::ifrt::MemoryKind()
             : xla::ifrt::MemoryKind(getMemoryKind().str());
};

//===----------------------------------------------------------------------===//
// IfrtUnspecifiedShardingAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtUnspecifiedShardingAttr::CanApplyTo(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, llvm::ArrayRef<int> device_ids) const {
  // The unspecified sharding can be applied to any array.
  return mlir::success();
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtUnspecifiedShardingAttr::GlobalShapeFromLocalShape(
    llvm::ArrayRef<int64_t> local_shape) const {
  // Unspecified sharding does not change the shape.
  llvm::SmallVector<int64_t> global_shape(local_shape.begin(),
                                          local_shape.end());
  return global_shape;
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtUnspecifiedShardingAttr::LocalShapeFromGlobalShape(
    llvm::ArrayRef<int64_t> global_shape) const {
  // Unspecified sharding does not change the shape.
  llvm::SmallVector<int64_t> local_shape(global_shape.begin(),
                                         global_shape.end());
  return local_shape;
}

int IfrtUnspecifiedShardingAttr::NumDevices() const { return 0; }

xla::ifrt::MemoryKind IfrtUnspecifiedShardingAttr::MemoryKind() const {
  return xla::ifrt::MemoryKind();
}

//===----------------------------------------------------------------------===//
// IfrtArrayType
//===----------------------------------------------------------------------===//

// Returns an array of logical device ids.
llvm::ArrayRef<int> IfrtArrayType::getDevices() const {
  return getDevicesAttr().getIds();
}

mlir::LogicalResult IfrtArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, IfrtShardingAttrInterface sharding_attr,
    IfrtDevicesAttr devices) {
  return sharding_attr.CanApplyTo(emitError, shape, devices.getIds());
}

//===----------------------------------------------------------------------===//
// IfrtDevicesAttr
//===----------------------------------------------------------------------===//

IfrtDevicesAttr::operator llvm::ArrayRef<int>() const { return getIds(); }

mlir::LogicalResult IfrtDevicesAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int> ids) {
  llvm::DenseSet<int> device_set;
  for (int id : ids) {
    if (id < 0) {
      return emitError() << "Device list has negative logical id " << id;
    }
    if (auto [unused_it, inserted] = device_set.insert(id); !inserted) {
      return emitError() << "Device list has duplicate logical id " << id;
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfrtIntervalAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtIntervalAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int start,
    int end, int step) {
  if (start < 0 || end < 0) {
    return emitError() << "start, end must be zero or positive";
  }
  if (step <= 0) {
    return emitError() << "step must be positive";
  }
  if (start > end) {
    return emitError() << "interval is empty";
  }
  return mlir::success();
}

int IfrtIntervalAttr::size() const {
  return (getEnd() - getStart() + getStep() - 1) / getStep();
}

//===----------------------------------------------------------------------===//
// IfrtMappingAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtMappingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    IfrtIntervalAttr from_shards, IfrtIntervalAttr to_shards) {
  // Verify that from and to contains the same number of shards.
  if (from_shards.size() != to_shards.size()) {
    return emitError() << "from has " << from_shards.size() << " and to has "
                       << to_shards.size()
                       << ", but they must have the same number of shards.";
  }
  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla
