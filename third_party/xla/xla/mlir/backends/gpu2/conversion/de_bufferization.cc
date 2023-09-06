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

#include "xla/mlir/backends/gpu2/conversion/de_bufferization.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string_view>

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

namespace xla::gpu {

using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

TypedValue<MemRefType> stripReinterpretCast(TypedValue<MemRefType> value) {
  if (auto op =
          dyn_cast_or_null<memref::ReinterpretCastOp>(value.getDefiningOp()))
    return cast<TypedValue<MemRefType>>(op.getSource());
  return value;
}

TypedValue<MemRefType> stripReinterpretCast(TypedValue<BaseMemRefType> value) {
  return stripReinterpretCast(cast<TypedValue<MemRefType>>(value));
}

//===----------------------------------------------------------------------===//
// DeBufferization implementation
//===----------------------------------------------------------------------===//

// Returns the length of a memref in bytes.
static int64_t getMemrefLength(MemRefType memref) {
  return memref.getNumElements() *
         std::max(8u, memref.getElementType().getIntOrFloatBitWidth()) / 8;
}

// Returns a slice inferred from the memref.view operation.
static FailureOr<DeBufferization::Slice> getMemrefViewSlice(memref::ViewOp op) {
  BlockArgument arg = cast<BlockArgument>(op.getSource());
  MemRefType memref = op.getType();

  llvm::APInt offset;
  if (!matchPattern(op.getByteShift(), m_ConstantInt(&offset)))
    return failure();

  return DeBufferization::Slice(arg, offset.getSExtValue(),
                                getMemrefLength(memref));
}

// Returns a function argument corresponding to the constant name.
//
// Example:
//
//   memref.global "private" constant @cst : memref<2x3xf32>
//   func @main(%arg0: memref<24xi8> {lmhlo.constant_name = "cst"})
//
static FailureOr<DeBufferization::Slice> getConstantArg(
    func::FuncOp func, std::string_view constant_name) {
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto cst = func.getArgAttrOfType<StringAttr>(i, "lmhlo.constant_name");
        cst && cst.getValue().equals(constant_name)) {
      return DeBufferization::Slice(
          func.getArgument(i), /*offset=*/0,
          getMemrefLength(cast<MemRefType>(func.getArgument(i).getType())));
    }
  }
  return failure();
}

DeBufferization::Slice::Slice(BlockArgument arg, int64_t offset, int64_t length)
    : Tuple(arg, offset, length) {}

/*static*/ FailureOr<DeBufferization> DeBufferization::Create(
    TypeConverter &type_converter, ModuleOp module) {
  DeBufferization state(type_converter);

  // Update `state` to track memref.view operation.
  auto track_view = [&](memref::ViewOp op) -> WalkResult {
    auto func = op->getParentOfType<func::FuncOp>();
    if (auto slice = getMemrefViewSlice(op); succeeded(slice)) {
      state.aliased_set_[*slice].push_back(op);
      state.memref_slice_.try_emplace(op.getResult(), *slice);
      if (func.getArgAttr(slice->arg().getArgNumber(), "lmhlo.output_index")) {
        state.outputs_[func].push_back(op.getResult());
      }
      return WalkResult::advance();
    }

    return op->emitError("failed to get function argument slice");
  };

  // Update state to track memref.get_global operation.
  auto track_global = [&](memref::GetGlobalOp op) -> WalkResult {
    auto func = op->getParentOfType<func::FuncOp>();
    if (auto slice = getConstantArg(func, op.getName()); succeeded(slice)) {
      state.aliased_set_[*slice].push_back(op);
      state.memref_slice_.try_emplace(op.getResult(), *slice);
      return WalkResult::advance();
    }

    return op->emitError("failed to get constant argument");
  };

  // Walk all operations that can construct new memrefs.
  if (module.walk(track_view).wasInterrupted() ||
      module.walk(track_global).wasInterrupted())
    return failure();

  return state;
}

std::optional<DeBufferization::Slice> DeBufferization::getSlice(
    TypedValue<MemRefType> memref) {
  if (auto it = memref_slice_.find(memref); it != memref_slice_.end())
    return it->second;
  return std::nullopt;
}

llvm::ArrayRef<TypedValue<MemRefType>> DeBufferization::getOutputs(
    func::FuncOp func) {
  return outputs_[func];
}

void DeBufferization::remap(Block *block, TypedValue<MemRefType> memref,
                            TypedValue<TensorType> tensor) {
  assert(block && memref && tensor && "values must be not null");
  assert(memref_slice_.contains(stripReinterpretCast(memref)));
  remapped_[block][memref_slice_[stripReinterpretCast(memref)]] = tensor;
}

mlir::TypedValue<mlir::TensorType> DeBufferization::remapped(
    mlir::Block *block, mlir::TypedValue<mlir::MemRefType> memref) {
  assert(block && memref && "memref must be not null");
  assert(memref_slice_.contains(stripReinterpretCast(memref)));
  return remapped_[block][memref_slice_[stripReinterpretCast(memref)]];
}

TypedValue<TensorType> DeBufferization::remapped(
    ImplicitLocOpBuilder &b, TypedValue<MemRefType> memref) {
  Type expected = type_converter_.convertType(memref.getType());

  auto tensor = remapped(b.getBlock(), memref);
  if (!tensor || tensor.getType() == expected) return tensor;

  // Cast a tensor to expected type via buffer export/import.
  auto export_op = b.create<IREE::Input::TensorExportOp>(
      b.getType<IREE::Input::BufferType>(), tensor, ValueRange());
  auto import_op =
      b.create<IREE::Input::TensorImportOp>(expected, export_op, ValueRange());

  // Update memref -> tensor remapping to the latest tensor value.
  tensor = cast<TypedValue<TensorType>>(import_op.getResult());
  remap(b.getBlock(), memref, tensor);
  return tensor;
}

//===----------------------------------------------------------------------===//
// Helper functions for de-bufferizing operatrions with nested regions
//===----------------------------------------------------------------------===//

using MemRefFilter = llvm::function_ref<bool(TypedValue<MemRefType>)>;

UsedBuffers getUsedBuffers(ArrayRef<Block *> blocks, MemRefFilter filter) {
  UsedBuffers buffers;

  // TODO(ezhulenev): Add support for all lmhlo and lmhlo_gpu operations.
  for (Block *block : blocks) {
    block->walk([&](bufferization::ToTensorOp op) {
      buffers.read.insert(stripReinterpretCast(op.getMemref()));
    });

    block->walk([&](memref::TensorStoreOp op) {
      buffers.write.insert(stripReinterpretCast(op.getMemref()));
    });

    block->walk([&](lmhlo::SortOp op) {
      for (auto input : op.getInputs())
        buffers.read.insert(
            stripReinterpretCast(cast<TypedValue<MemRefType>>(input)));
      for (auto output : op.getOutput())
        buffers.write.insert(
            stripReinterpretCast(cast<TypedValue<MemRefType>>(output)));
    });
  }

  // Discard filtered-out read/write buffers.
  assert(filter && "filter must be not null");
  auto discard = [&](auto memref) { return filter(memref) == false; };
  buffers.read.remove_if(discard);
  buffers.write.remove_if(discard);

  // Remove written buffers from read buffers.
  buffers.read.remove_if(
      [&](auto memref) { return buffers.write.contains(memref); });

  return buffers;
}

UsedBuffers getUsedBuffers(llvm::ArrayRef<Block *> blocks) {
  return getUsedBuffers(blocks, [](auto) { return true; });
}

}  // namespace xla::gpu
