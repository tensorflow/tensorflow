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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_DE_BUFFERIZATION_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_DE_BUFFERIZATION_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace xla {
namespace gpu {

// As a part of the compilation pipeline to prepare XLA executable to run on top
// of IREE VM we convert it from LMHLO dialects to IREEInput dialect. The key
// difference is that at LMHLO level program operates on buffers (memrefs),
// and at IREEInput level it uses value semantics and tensors.
//
// We rely on in-place semantics of IREEInput operations (tied operands) to
// re-write operations writing to buffers to operations updating tensors.
//
// Example:
//
//   func @main(%arg0: memref<f32>, %arg1: memref<f32>) {
//     lmhlo.fusion {
//       %0 = bufferization.to_tensor %arg0 : memref<f32>
//       %0 = bufferization.to_tensor %arg1 : memref<f32>
//       %2 = mhlo.add %0, %1: tensor<f32>
//       memref.tensor_store %2, %arg1 : memref<f32>
//     }
//     "some.consumer"(%arg1) : (memref<f32>) -> ()
//   }
//
// In this example `%arg0` is a read only buffer, and `%arg1` is a read-write
// buffer.
//
//  func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) {
//     %0 = iree_input.dispatch @add(%arg0, %arg1)
//       : (tensor<f32>, tensor<f32>) -> %arg1
//     "some.consumer"(%0) : (tensor<f32>) -> ()
//   }
//
// We use `DeBufferization` to track the mapping from a memref to the last
// tensor produced by an operation that wrote into a memref. In the example
// above instead of passing `%arg1` to the consumer, we pass the last tensor
// that shares the storage with `%arg1`. After lowering this representation to
// IREEs HAL dialect, it's guaranteed that XLA program will read/write from/to
// exactly the same buffer slices as its original version.
//
// In XLA all input arguments do not alias, so we don't need any buffer aliasing
// analysis, and we can safely rely on memref.view operation offsets and sizes.
//
// Conversion implementation is a bit more complicated because we have to handle
// memref.view and memref.reinterpret_cast operations, but all conversions
// conceptually are doing the same transformation as in example above.
struct DeBufferization {
  // Mapping block block arguments to memref views constructed from them. We'll
  // need it at the very end to tie all inplace updates to the optimization
  // barrier to prevent dead code elimination.
  llvm::DenseMap<mlir::BlockArgument,
                 llvm::SmallVector<mlir::TypedValue<mlir::MemRefType>>>
      imported;

  // Mapping from the memref view to the last tensor that is tied to the same
  // underlying storage. We use this mapping to thread inplace tensor updates
  // through all operations in the compiled function.
  llvm::DenseMap<mlir::Block *,
                 llvm::DenseMap<mlir::TypedValue<mlir::MemRefType>,
                                mlir::TypedValue<mlir::TensorType>>>
      remapped;
};

// We only pass around tensors constructed from a row major memrefs because
// currently IREE buffer view can't represent a strided layout. As a short term
// solution the plan is to pass tensor layout as a side data structure, but
// longer term we'll need to add tensor/buffer layouts to IREE HAL buffers.
mlir::TypedValue<mlir::MemRefType> stripReinterpretCast(
    mlir::TypedValue<mlir::MemRefType> value);

mlir::TypedValue<mlir::MemRefType> stripReinterpretCast(
    mlir::TypedValue<mlir::BaseMemRefType> value);

//===----------------------------------------------------------------------===//
// Helper functions for de-bufferizing operations with nested regions
//===----------------------------------------------------------------------===//

struct UsedBuffers {
  llvm::SetVector<mlir::TypedValue<mlir::MemRefType>> read;
  llvm::SetVector<mlir::TypedValue<mlir::MemRefType>> write;
};

UsedBuffers getUsedBuffers(llvm::ArrayRef<mlir::Block *> blocks);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_DE_BUFFERIZATION_H_
