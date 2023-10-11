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

#ifndef XLA_MLIR_BACKENDS_GPU2_CONVERSION_DE_BUFFERIZATION_H_
#define XLA_MLIR_BACKENDS_GPU2_CONVERSION_DE_BUFFERIZATION_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace xla {
namespace gpu {

// As a part of the compilation pipeline to prepare XLA executable to run on top
// of IREE VM we convert it from LMHLO dialects to IREEInput dialect. The key
// difference is that at LMHLO level program operates on buffers (memrefs),
// and at IREEInput level it uses value semantics and tensors.
//
// The goal of this conversion is to materialize HLO schedule and buffer
// assignment in a SSA form, so that we can run further optimizations on it.
//
// We rely on in-place semantics of IREEInput operations (tied operands) to
// re-write operations writing to buffers to operations updating tensors.
//
// Example:
//
//   func @main(%arg0: memref<f32>, %arg1: memref<f32>) {
//     lmhlo.fusion {
//       %0 = bufferization.to_tensor %arg0 : memref<f32>
//       %1 = bufferization.to_tensor %arg1 : memref<f32>
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
// In XLA all input arguments (entry point function arguments) do not alias, so
// we don't need any complex buffer aliasing analysis, however memrefs created
// with memref.view operations can alias, and for that we track aliasing to
// guarantee correct scheduling (see details below).
//
// Conversion implementation is a bit more complicated because we have to handle
// memref.view and memref.reinterpret_cast operations, but all conversions
// conceptually are doing the same transformation as in example above.
class DeBufferization {
 public:
  // Initialize de-bufferization internal state from a given module.
  static mlir::FailureOr<DeBufferization> Create(
      mlir::TypeConverter &type_converter, mlir::ModuleOp module);

  // In XLA multiple memrefs of different type can share the underlying storage
  // at different points in the input program, because when we convert from HLO
  // to LMHLO we do not materialize bitcasts when we pass results of one
  // operation to arguments of a second one.
  //
  // Furthermore unrelated operations can also share a storage at different
  // points in the XLA program. Conceptually you can think of the buffer slice
  // (memref taken from one of the arguments) as a register that can hold
  // different values at different points (albeit a weird one that can alias
  // with other registers).
  //
  // Example: We rely on i32 -> ui32 bitcast to be a no-op.
  //
  //   %view_0 = memref.view %arg0[%c0] : memref<4xi8> to memref<i32>
  //   lmhlo.fusion @fusion_0 {
  //     memref.tensor_store %0, %view_0
  //   }
  //   ...
  //   %view_1 = memref.view %arg0[%c0] : memref<4xi8> to memref<ui32>
  //   lmhlo.fusion @fusion_1 {
  //     bufferization.to_tensor %view_1 : memref<ui32>
  //   }
  //
  // To guarantee that we do not schedule @fusion_0 concurrently with @fusion_1
  // we need to track memrefs sharing the underlying storage, and thread all
  // tensors sharing the storage using tied operands.
  //
  // The correct lowering should look like this:
  //
  //    %tensor_0 = ... : tensor<i32>
  //    %0 = dispatch @fusion_0 (%tensor_0 : tensor<i32>) -> %tensor_0
  //    %tensor_1 = <cast tensor %0> : tensor<ui32>
  //    %1 = dispatch @fusion_1 (%tensor_1 : tensor<ui32>) -> ...
  //
  // TODO(ezhulenev): Today we only track "full aliasing", however for example
  // concat operation can (can it really?) introduce partial aliasing between
  // buffer subspans, and to correctly handle that we need to add "barriers" to
  // IR that will enforce ordering via tied operands. This is work in progress,
  // but should be doable with streamable calls to external module, that will be
  // a no-op at run time.
  struct Slice : public std::tuple<mlir::BlockArgument, int64_t, int64_t> {
    using Tuple = std::tuple<mlir::BlockArgument, int64_t, int64_t>;
    using Tuple::Tuple;

    Slice(Tuple base) : Tuple(base) {}  // NOLINT
    Slice(mlir::BlockArgument arg, int64_t offset, int64_t length);

    mlir::BlockArgument arg() { return std::get<0>(*this); }
    int64_t offset() { return std::get<1>(*this); }
    int64_t length() { return std::get<2>(*this); }
  };

  //===--------------------------------------------------------------------===//
  // Tracking of constructed memrefs.
  //===--------------------------------------------------------------------===//

  // Returns an argument slice for a given memref.
  std::optional<Slice> getSlice(mlir::TypedValue<mlir::MemRefType> memref);

  // Return memrefs constructed from function output arguments.
  llvm::ArrayRef<mlir::TypedValue<mlir::MemRefType>> getOutputs(
      mlir::func::FuncOp func);

  //===--------------------------------------------------------------------===//
  // Tracking of memref -> tensor mapping.
  //===--------------------------------------------------------------------===//

  // Remaps `memref` to a `tensor` inside a `block`.
  void remap(mlir::Block *block, mlir::TypedValue<mlir::MemRefType> memref,
             mlir::TypedValue<mlir::TensorType> tensor);

  // Returns `memref` remapping to a tensor inside a `block`. Returned tensor
  // type might not be compatible with memref type. Use function declared below
  // to automatically insert type casting
  mlir::TypedValue<mlir::TensorType> remapped(
      mlir::Block *block, mlir::TypedValue<mlir::MemRefType> memref);

  // Returns `memref` remapping to a tensor at `b` insertion point. Might create
  // new IR to cast between different tensor types aliasing the slice. Returns
  // nullptr if memref has no remapping.
  mlir::TypedValue<mlir::TensorType> remapped(
      mlir::ImplicitLocOpBuilder &b, mlir::TypedValue<mlir::MemRefType> memref);

 private:
  explicit DeBufferization(mlir::TypeConverter &type_converter)
      : type_converter_(type_converter) {}

  mlir::TypeConverter &type_converter_;

  // A mapping from a function to memrefs constructed from output arguments.
  llvm::DenseMap<mlir::func::FuncOp,
                 llvm::SmallVector<mlir::TypedValue<mlir::MemRefType>>>
      outputs_;

  // A reverse mapping from a memref to the corresponding argument slice.
  llvm::DenseMap<mlir::TypedValue<mlir::MemRefType>, Slice> memref_slice_;

  // A set of memrefs aliasing an argument slice. We'll use this information to
  // automatically cast between different views of the same underlying buffer.
  llvm::DenseMap<Slice, llvm::SmallVector<mlir::TypedValue<mlir::MemRefType>>>
      aliased_set_;

  // Mapping from the slice to the last tensor that is tied to it in the block.
  // We use this mapping to thread inplace tensor updates through all operations
  // in the compiled function.
  llvm::DenseMap<mlir::Block *,
                 llvm::DenseMap<Slice, mlir::TypedValue<mlir::TensorType>>>
      remapped_;
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

// Returns all buffers used in the given blocks.
UsedBuffers getUsedBuffers(llvm::ArrayRef<mlir::Block *> blocks);

// Returns buffers used in the given blocks and for which predicate is `true`.
UsedBuffers getUsedBuffers(
    llvm::ArrayRef<mlir::Block *> blocks,
    llvm::function_ref<bool(mlir::TypedValue<mlir::MemRefType>)> filter);

}  // namespace gpu
}  // namespace xla

namespace llvm {

template <>
struct DenseMapInfo<xla::gpu::DeBufferization::Slice> {
  using Slice = xla::gpu::DeBufferization::Slice;

  static Slice getEmptyKey() {
    return DenseMapInfo<Slice::Tuple>::getEmptyKey();
  }
  static Slice getTombstoneKey() {
    return DenseMapInfo<Slice::Tuple>::getTombstoneKey();
  }
  static unsigned getHashValue(const Slice &value) {
    return DenseMapInfo<Slice::Tuple>::getHashValue(value);
  }
  static bool isEqual(const Slice &a, const Slice &b) {
    return DenseMapInfo<Slice::Tuple>::isEqual(a, b);
  }
};

}  // namespace llvm

#endif  // XLA_MLIR_BACKENDS_GPU2_CONVERSION_DE_BUFFERIZATION_H_
