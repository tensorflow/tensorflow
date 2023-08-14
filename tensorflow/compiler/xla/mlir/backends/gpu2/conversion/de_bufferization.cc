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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

namespace xla::gpu {

using namespace mlir;  // NOLINT

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
// Helper functions for de-bufferizing operatrions with nested regions
//===----------------------------------------------------------------------===//

UsedBuffers getUsedBuffers(ArrayRef<Block *> blocks) {
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

  // Remove written buffers from read buffers.
  buffers.read.remove_if(
      [&](auto memref) { return buffers.write.contains(memref); });

  return buffers;
}

}  // namespace xla::gpu
