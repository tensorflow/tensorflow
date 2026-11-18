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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/strings/str_format.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/tpu_ops_impl.h"  // IWYU pragma: keep.
#include "xla/mosaic/dialect/tpu/util.h"
#include "xla/mosaic/util.h"

namespace mlir {
namespace tpu {

namespace {

std::optional<CoreType> getRefCoreType(TypedValue<MemRefType> value) {
  if (!value) {
    return std::nullopt;
  }
  auto space = dyn_cast_if_present<tpu::MemorySpaceAttr>(
      value.getType().getMemorySpace());
  if (!space) {
    return std::nullopt;
  }
  return space.getCoreType();
}

}  // namespace

LogicalResult SharedMemRefSliceOp::verify() {
  CoreType core_type = GetCoreTypeOfParentOp(**this);
  if (core_type != CoreType::kScVectorSubcore) {
    return emitOpError("Unsupported core type: ") << core_type;
  }
  return verifySharedMemRefSliceOp(
      *this, MemorySpaceAttr::get(getContext(), MemorySpace::kVmem,
                                  CoreType::kScVectorSubcore));
}

void VectorStoreOp::build(OpBuilder& builder, OperationState& state,
                          Value valueToStore, Value base, ValueRange indices,
                          Value mask, bool add) {
  build(builder, state, valueToStore, base, indices,
        /*strides=*/builder.getDenseI32ArrayAttr({}), mask, add);
}

void VectorLoadOp::build(OpBuilder& builder, OperationState& state,
                         Type result_type, Value base, ValueRange indices,
                         Value mask) {
  build(builder, state, result_type, base, indices,
        /*strides=*/builder.getDenseI32ArrayAttr({}), mask);
}

namespace {

CoreType getSemaphoreSignalTargetCoreType(SemaphoreSignalOp op) {
  return getRefCoreType(op.getSemaphore()).value_or(GetCoreTypeOfParentOp(*op));
}

}  // namespace

void SemaphoreSignalOp::build(OpBuilder& builder, OperationState& state,
                              Value semaphore, Value amount, Value device_id,
                              Value core_id) {
  build(builder, state, semaphore, amount, device_id, core_id,
        /*subcore_id=*/nullptr);
}

LogicalResult SemaphoreSignalOp::verify() {
  RETURN_IF_FAILED(verifySemaphoreSignalOp(*this));

  CoreType issuing_core_type = GetCoreTypeOfParentOp(**this);
  CoreType target_core_type = getSemaphoreSignalTargetCoreType(*this);

  if (!isRemote(*this)) {
    if (target_core_type != issuing_core_type) {
      return emitOpError(
          absl::StrFormat("Target core type (%s) must match source core type "
                          "(%s) when device_id and core_id are not specified",
                          stringifyCoreType(target_core_type),
                          stringifyCoreType(issuing_core_type)));
    }
  }
  // Subcore ID applies only to SC vector subcore semaphore ops.
  if (target_core_type != CoreType::kScVectorSubcore &&
      getSubcoreId() != nullptr) {
    return emitOpError(
        "Subcore id should not be set unless target core type is SC vector "
        "subcore");
  }
  return success();
}

namespace {

bool isSparseCoreStreamLocalMemory(tpu::MemorySpaceAttr mem_space,
                                   CoreType issuing_core) {
  if (mem_space.getCoreType().value_or(issuing_core) != issuing_core) {
    return false;
  }
  if (issuing_core == CoreType::kScVectorSubcore) {
    return mem_space.getValue() == MemorySpace::kVmem;
  }
  if (issuing_core == CoreType::kScScalarSubcore) {
    return mem_space.getValue() == MemorySpace::kSmem;
  }
  return false;
}

LogicalResult verifySparseCoreDmaSemaphores(
    Operation* op, tpu::MemorySpaceAttr source_mem_space,
    tpu::MemorySpaceAttr target_mem_space, Value source_semaphore,
    Value target_semaphore, bool is_remote, CoreType issuing_core) {
  if (is_remote) {
    return success();
  }
  bool src_is_local =
      isSparseCoreStreamLocalMemory(source_mem_space, issuing_core);
  bool tgt_is_local =
      isSparseCoreStreamLocalMemory(target_mem_space, issuing_core);
  if (!src_is_local && tgt_is_local) {
    if (source_semaphore != nullptr) {
      return op->emitOpError(
          "Source semaphores are unsupported for transfers from a "
          "non-local memory to local memory on SparseCore");
    }
  }
  return success();
}

LogicalResult verifyCommonDmaCoreParams(Operation* op, MemRefType source_ty,
                                        MemRefType target_ty,
                                        TypedValue<MemRefType> source_sem,
                                        TypedValue<MemRefType> target_sem,
                                        Value core_id, Value subcore_id,
                                        bool strict_ordering,
                                        CoreType issuing_core) {
  bool is_sc = issuing_core == CoreType::kScVectorSubcore ||
               issuing_core == CoreType::kScScalarSubcore;
  if (source_sem &&
      getRefCoreType(source_sem).value_or(issuing_core) != issuing_core) {
    return op->emitOpError(
        "Source semaphore and source ref core type mismatched");
  }
  // If the target core_type is different from the issuing core_type,
  // the specific core_id must be provided. The device_id is irrelevant here.
  CoreType target_core = getRefCoreType(target_sem).value_or(issuing_core);
  if (target_core != issuing_core && core_id == nullptr) {
    return op->emitOpError(
        absl::StrFormat("Core id must be specified when target core type (%v) "
                        "is different from source core type (%v)",
                        target_core, issuing_core));
  }
  if (strict_ordering && !is_sc) {
    return op->emitOpError(
        "Strict ordering is only supported on the SC scalar and vector "
        "subcores");
  }
  if (Value sem = target_sem != nullptr ? target_sem : source_sem;
      sem != nullptr &&
      isa<SemaphoreType>(getMemRefType(sem).getElementType())) {
    if (HasMemorySpace(source_ty, MemorySpace::kSmem, CoreType::kTc) ||
        HasMemorySpace(target_ty, MemorySpace::kSmem, CoreType::kTc)) {
      return op->emitOpError(
          "Non-DMA semaphores are not supported for DMAs involving SMEM");
    }
  }
  // Subcore ID applies only to SC vector subcore DMAs.
  if (target_core != CoreType::kScVectorSubcore && subcore_id != nullptr) {
    return op->emitOpError(
        "Subcore id should not be set unless target core type is SC vector "
        "subcore");
  }
  return success();
}

}  // namespace

void EnqueueDMAOp::build(OpBuilder& builder, OperationState& state,
                         Value source, Value source_semaphore, Value target,
                         Value target_semaphore, Value device_id, Value core_id,
                         uint32_t priority, bool strict_ordering) {
  build(builder, state, source, source_semaphore, target, target_semaphore,
        device_id, core_id, /*subcore_id=*/nullptr, priority, strict_ordering);
}

LogicalResult EnqueueDMAOp::verify() {
  MemRefType source_ty = getMemRefType(getSource());
  MemRefType target_ty = getMemRefType(getTarget());
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  bool is_remote = isRemote(*this);
  bool is_sc = issuing_core == CoreType::kScVectorSubcore ||
               issuing_core == CoreType::kScScalarSubcore;

  RETURN_IF_FAILED(verifyEnqueueDMAOp(*this, is_sc));
  RETURN_IF_FAILED(verifyCommonDmaCoreParams(
      getOperation(), source_ty, target_ty, getSourceSemaphore(),
      getTargetSemaphore(), getCoreId(), getSubcoreId(), getStrictOrdering(),
      issuing_core));

  if (is_remote) {
    if (!getSourceSemaphore()) {
      return emitOpError(
          "DMA source semaphore must be specified when device_id or core_id is "
          "specified");
    }
  }

  if (is_sc) {
    auto source_mem_space =
        dyn_cast_or_null<tpu::MemorySpaceAttr>(source_ty.getMemorySpace());
    auto target_mem_space =
        dyn_cast_or_null<tpu::MemorySpaceAttr>(target_ty.getMemorySpace());
    if (source_mem_space != nullptr && target_mem_space != nullptr) {
      RETURN_IF_FAILED(verifySparseCoreDmaSemaphores(
          getOperation(), source_mem_space, target_mem_space,
          getSourceSemaphore(), getTargetSemaphore(), is_remote, issuing_core));
    }
  } else {
    if (getTargetSemaphore() == nullptr) {
      return emitOpError("DMA target semaphore must be specified");
    }
    if (!is_remote && getSourceSemaphore() != nullptr) {
      return emitOpError(
          "DMA destination device_id or core_id must be specified when source "
          "semaphore is specified");
    }
  }
  return success();
}

LogicalResult EnqueueIndirectDMAOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (issuing_core != CoreType::kScVectorSubcore) {
    return emitOpError(
        "Enqueue indirect DMA is supported only on the SC vector subcore");
  }
  return verifyEnqueueIndirectDMAOp(*this);
}

void WaitDMA2Op::build(OpBuilder& builder, OperationState& state,
                       Value semaphore, Value src, Value dst) {
  build(builder, state, semaphore, src, dst, /*device_id=*/nullptr,
        /*core_id=*/nullptr);
}

LogicalResult WaitDMA2Op::verify() {
  RETURN_IF_FAILED(verifyWaitDMA2Op(*this));

  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (getRefCoreType(getSemaphore()).value_or(issuing_core) != issuing_core) {
    return emitOpError("Can only await semaphores attached to the local core");
  }
  return success();
}

LogicalResult WaitDMAOp::verify() {
  MemRefType source_ty = getMemRefType(getSource());
  MemRefType target_ty = getMemRefType(getTarget());
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  bool is_remote = isRemote(*this);
  bool is_sc = issuing_core == CoreType::kScVectorSubcore ||
               issuing_core == CoreType::kScScalarSubcore;

  RETURN_IF_FAILED(verifyWaitDMAOp(*this, is_sc));
  RETURN_IF_FAILED(verifyCommonDmaCoreParams(
      getOperation(), source_ty, target_ty, getSourceSemaphore(),
      getTargetSemaphore(), getCoreId(), getSubcoreId(), getStrictOrdering(),
      issuing_core));

  bool wait_destination = getWaitTarget();
  TypedValue<MemRefType> sem =
      wait_destination ? getTargetSemaphore() : getSourceSemaphore();

  if (is_sc) {
    if (!is_remote) {
      auto source_mem_space =
          dyn_cast_or_null<tpu::MemorySpaceAttr>(source_ty.getMemorySpace());
      auto target_mem_space =
          dyn_cast_or_null<tpu::MemorySpaceAttr>(target_ty.getMemorySpace());
      if (source_mem_space != nullptr && target_mem_space != nullptr) {
        bool src_is_local =
            isSparseCoreStreamLocalMemory(source_mem_space, issuing_core);
        bool tgt_is_local =
            isSparseCoreStreamLocalMemory(target_mem_space, issuing_core);
        if (!src_is_local && tgt_is_local && !wait_destination) {
          return emitOpError(
              "Awaiting source read completion is unsupported for transfers "
              "from a non-local memory to local memory on SparseCore");
        }
      }
    }
  } else {
    if (wait_destination) {
      if (getTargetSemaphore() == nullptr) {
        return emitOpError("DMA target semaphore must be specified");
      }
      if (!is_remote && getSourceSemaphore() != nullptr) {
        return emitOpError(
            "DMA destination device_id or core_id must be specified when "
            "source semaphore is specified");
      }
    } else {
      if (getSourceSemaphore() == nullptr) {
        return emitOpError("The awaited semaphore must be provided");
      }
    }
  }
  if (sem) {
    if (getRefCoreType(sem).value_or(issuing_core) != issuing_core) {
      return emitOpError(
          "Can only await semaphores attached to the local core");
    }
  }
  return success();
}

LogicalResult WaitIndirectDMAOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (issuing_core != CoreType::kScVectorSubcore) {
    return emitOpError(
        "Wait indirect DMA is supported only on the SC vector subcore");
  }
  return verifyWaitIndirectDMAOp(*this);
}

LogicalResult LogOp::verify() {
  CoreType logging_core = GetCoreTypeOfParentOp(**this);
  bool is_sc_core = logging_core == CoreType::kScScalarSubcore ||
                    logging_core == CoreType::kScVectorSubcore;
  if (is_sc_core && getFormattedAttr() != nullptr &&
      getFormattedAttr().getValue()) {
    return emitOpError("Formatted logging is not supported on SC");
  }
  if (is_sc_core && getInputs().size() > 1) {
    return emitOpError("SC logging only supports 0 or 1 inputs");
  }
  if (logging_core == CoreType::kScScalarSubcore) {
    for (mlir::Value input : getInputs()) {
      if (llvm::isa<VectorType>(input.getType())) {
        return emitOpError(
            "SC scalar subcore does not support logging vectors");
      }
    }
  }
  return success();
}

void PackMaskOp::build(OpBuilder& builder, OperationState& state,
                       const VectorType output_type,
                       const ArrayRef<Value> padded_sources) {
  SmallVector<Value> sources;
  SmallVector<int32_t> positions;
  for (size_t i = 0; i < padded_sources.size(); ++i) {
    if (padded_sources[i] != nullptr) {
      sources.push_back(padded_sources[i]);
      positions.push_back(i);
    }
  }
  build(builder, state, output_type, sources, positions);
}

LogicalResult FetchAndAddSyncOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  CoreType target_core = getRefCoreType(getBase()).value_or(issuing_core);
  switch (target_core) {
    case CoreType::kScVectorSubcore:
      break;
    case CoreType::kScScalarSubcore:
      return emitOpError("SC scalar subcore target not supported yet");
    default:
      return emitOpError("Only SC vector subcore is supported, got ")
             << target_core;
  }
  return verifyFetchAndAddSyncOp(*this);
}

LogicalResult CoreIdOp::verify() {
  Operation* parent = GetParentOpWithCoreType(**this);
  if (parent == nullptr) {
    return emitOpError("Failed to infer the core type from parent ops");
  }
  CoreType core_type = *TPUDialect::GetCoreTypeAttr(parent);
  if (core_type == CoreType::kTc) {
    return emitOpError("Unsupported core type: ") << core_type;
  }
  // Querying Core ID is not supported under dimension semantics.
  if (auto func_op = getOperation()->getParentOfType<func::FuncOp>();
      func_op != nullptr &&
      func_op->getAttrOfType<ArrayAttr>("dimension_semantics") != nullptr) {
    return failure();
  }
  return success();
}

LogicalResult SubcoreIdOp::verify() {
  Operation* parent = GetParentOpWithCoreType(**this);
  if (parent == nullptr) {
    return emitOpError("Failed to infer the core type from parent ops");
  }
  CoreType core_type = *TPUDialect::GetCoreTypeAttr(parent);
  if (core_type == CoreType::kTc) {
    return emitOpError("Unsupported core type: ") << core_type;
  }
  // Subcore ID is not supported under dimension semantics.
  if (auto func_op = getOperation()->getParentOfType<func::FuncOp>();
      func_op != nullptr &&
      func_op->getAttrOfType<ArrayAttr>("dimension_semantics") != nullptr) {
    return failure();
  }
  return success();
}

}  // namespace tpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_ops.cc.inc"
