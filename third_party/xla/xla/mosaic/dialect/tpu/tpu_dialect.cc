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

#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep.
#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep.
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep.
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.cc.inc"
#include "xla/mosaic/dialect/tpu/tpu_dialect_impl.h"  // IWYU pragma: keep.
#include "xla/mosaic/dialect/tpu/tpu_enums.cc.inc"

// This is a bit unclean, but we need to squat the xla namespace to make sure
// that this overload is found via argument-dependent lookup.
namespace xla {

llvm::hash_code hash_value(const ::xla::Tile& p) { return absl::HashOf(p); }

}  // namespace xla

#define GET_ATTRDEF_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_attr_defs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_type_defs.cc.inc"

namespace mlir::tpu {

void TPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/mosaic/dialect/tpu/tpu_attr_defs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/mosaic/dialect/tpu/tpu_type_defs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/mosaic/dialect/tpu/tpu_ops.cc.inc"
      >();
}

/* static */ std::optional<CoreType> TPUDialect::GetCoreTypeAttr(
    Operation* op) {
  Attribute attr = op->getAttr(GetCoreTypeKey());
  if (attr == nullptr) {
    // For backwards compatibility we assume that the "main" function without
    // an explicit core type belongs to TensorCore.
    // TODO(b/505757864): Remove this in 6 months.
    if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(op);
        func_op && func_op.getName() == "main") {
      return CoreType::kTc;
    }
    return std::nullopt;
  }
  if (!mlir::isa<CoreTypeAttr>(attr)) {
    return std::nullopt;
  }
  return mlir::cast<CoreTypeAttr>(attr).getValue();
}

LogicalResult TPUDialect::verifyOperationAttribute(Operation* const op,
                                                   NamedAttribute attr) {
  CHECK(op != nullptr);
  if (attr.getName() == GetMainKey()) {
    if (!isa<func::FuncOp>(op) || !isa<mlir::UnitAttr>(attr.getValue())) {
      return op->emitOpError() << GetMainKey()
                               << " attribute is expected to be a unit "
                                  "attribute on a func.func operation";
    }
  }
  return success();
}

Operation* GetParentOpWithCoreType(Operation& op) {
  Operation* parent = &op;
  while ((parent = parent->getParentOp())) {
    if (auto core_type = TPUDialect::GetCoreTypeAttr(parent);
        core_type.has_value()) {
      return parent;
    }
  }
  return nullptr;
}

CoreType GetCoreTypeOfParentOp(Operation& op) {
  Operation* parent = GetParentOpWithCoreType(op);
  return parent ? *TPUDialect::GetCoreTypeAttr(parent) : CoreType::kTc;
}

absl::StatusOr<func::FuncOp> GetMainFunc(ModuleOp module, CoreType core_type) {
  func::FuncOp result = nullptr;
  for (auto func_op : module.getOps<func::FuncOp>()) {
    if (!func_op->hasAttr(TPUDialect::GetMainKey())) {
      continue;
    }
    if (TPUDialect::GetCoreTypeAttr(func_op) != core_type) {
      continue;
    }
    if (result != nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Multiple functions with %v attribute and tpu.core_type = %v found",
          TPUDialect::GetMainKey(), core_type));
    }
    result = func_op;
  }
  if (result != nullptr) {
    return result;
  }
  // If there is no function marked with tpu.main, fall back to finding a
  // (unique) function with the requested core type.
  for (func::FuncOp func_op : module.getOps<func::FuncOp>()) {
    if (TPUDialect::GetCoreTypeAttr(func_op) != core_type) {
      continue;
    }
    if (result != nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Multiple functions with tpu.core_type = %v found", core_type));
    }
    result = func_op;
  }
  if (result != nullptr) {
    return result;
  }
  // We have to maintain backwards compatibility with TC kernels which do not
  // set the tpu.core_type attribute.
  bool fallback_to_main = core_type == CoreType::kTc;
  if (!fallback_to_main) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "No function with tpu.core_type = %v found", core_type));
  }
  if (auto main_func = module.lookupSymbol<func::FuncOp>("main")) {
    return main_func;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "No function with tpu.core_type = %v nor a main function found",
      core_type));
}

TiledLayoutAttr TiledLayoutAttr::getContiguous(MLIRContext* context,
                                               ArrayRef<xla::Tile> tiles,
                                               ArrayRef<int64_t> shape) {
  return TiledLayoutAttr::get(context, tiles,
                              getContiguousTileStrides(tiles, shape));
}

SmallVector<int64_t> TiledLayoutAttr::getExpandedStrides() const {
  auto strides = mlir::tpu::getExpandedStrides(getTiles(), getTileStrides());
  CHECK(succeeded(strides));
  return *strides;
}

bool TiledLayoutAttr::hasDynamicStrides() const {
  return llvm::any_of(getTileStrides(), ShapedType::isDynamic);
}

int64_t TiledLayoutAttr::getNumDynamicStrides() const {
  return llvm::count_if(getTileStrides(), ShapedType::isDynamic);
}

LogicalResult MemorySpaceAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, MemorySpace value,
    std::optional<CoreType> core_type) {
  if (!core_type.has_value()) {
    return success();
  }
  switch (value) {
    case MemorySpace::kAny:
    case MemorySpace::kHbm:
    case MemorySpace::kHost:
    case MemorySpace::kVmemShared:
      return emitError() << "Memory space " << value
                         << " cannot be owned by a core";
    case MemorySpace::kVmem:
      if (*core_type == CoreType::kScScalarSubcore) {
        return emitError() << "Memory space " << value
                           << " cannot be owned by core " << *core_type;
      }
      return success();
    case MemorySpace::kCmem:
      if (*core_type != CoreType::kTc) {
        return emitError() << "Memory space " << value
                           << " cannot be owned by core " << *core_type;
      }
      return success();
    case MemorySpace::kSmem:
    case MemorySpace::kSemaphoreMem:
      return success();
  }
}

MemRefType getMemRefType(Value value) {
  if (auto erase_op = value.getDefiningOp<tpu::EraseLayoutOp>()) {
    value = erase_op.getOperand();
  }
  return cast<MemRefType>(value.getType());
}

DotDimensionNumbersAttr defaultDimensionNumbers(Builder& builder,
                                                bool transpose_lhs,
                                                bool transpose_rhs) {
  return tpu::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhs_contracting_dims=*/{transpose_lhs ? 0 : 1},
      /*rhs_contracting_dims=*/{transpose_rhs ? 1 : 0},
      /*lhs_non_contracting_dims=*/{transpose_lhs ? 1 : 0},
      /*rhs_non_contracting_dims=*/{transpose_rhs ? 0 : 1},
      /*output_dim_order=*/{0, transpose_lhs ? 1 : 0, 1, transpose_rhs ? 0 : 1},
      /*lhs_batch_dims=*/{},
      /*rhs_batch_dims=*/{});
}

const ::llvm::fltSemantics& Float8EXMYType::getFloatSemantics() const {
  return getUnderlyingType().getFloatSemantics();
}

namespace {

struct CommsAnalysisState {
  bool has_communication = false;
  bool has_custom_barrier = false;

  explicit operator bool() { return has_communication && has_custom_barrier; }
};

void analyzeCrossChipCommunication(mlir::Operation* op,
                                   CommsAnalysisState* state) {
  if (auto dma = dyn_cast<tpu::EnqueueDMAOp>(op)) {
    state->has_communication |= dma.getDeviceId() != nullptr;
  } else if (auto signal = dyn_cast<tpu::SemaphoreSignalOp>(op)) {
    state->has_communication |= signal.getDeviceId() != nullptr;
  } else if (auto barrier = dyn_cast<tpu::GetBarrierSemaphoreOp>(op)) {
    state->has_custom_barrier = true;
  }
  for (Region& region : op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        analyzeCrossChipCommunication(&op, state);
        if (*state) {
          return;
        }
      }
    }
  }
}

}  // namespace

std::pair<bool, bool> mightCommunicateBetweenChips(mlir::Operation* op) {
  CommsAnalysisState state;
  analyzeCrossChipCommunication(op, &state);
  return std::make_pair(state.has_communication, state.has_custom_barrier);
}

}  // namespace mlir::tpu
