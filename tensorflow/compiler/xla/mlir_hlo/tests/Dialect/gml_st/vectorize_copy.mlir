// RUN: mlir-hlo-opt %s --vectorize-copy --split-input-file | FileCheck %s

func.func @vectorize_copy(%arg: memref<10x16xf32>) -> memref<10x10xf32> {
  %subview = memref.subview %arg[0, 0] [10, 10] [1, 1] : memref<10x16xf32> to memref<10x10xf32, strided<[16, 1]>>
  %alloc = memref.alloc() : memref<10x10xf32>
  memref.copy %subview, %alloc : memref<10x10xf32, strided<[16, 1]>> to memref<10x10xf32>
  return %alloc : memref<10x10xf32>
}

// CHECK-LABEL: func @vectorize_copy

// CHECK-NOT:     memref.copy
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_write

// -----

func.func @do_not_vectorize_copy(%arg: memref<10x10xf32>) -> memref<10x10xf32> {
  %alloc_10 = memref.alloc() : memref<10x10xf32>
  memref.copy %arg, %alloc_10 : memref<10x10xf32> to memref<10x10xf32>
  return %alloc_10 : memref<10x10xf32>
}

// CHECK-LABEL: func @do_not_vectorize_copy

// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         memref.copy
