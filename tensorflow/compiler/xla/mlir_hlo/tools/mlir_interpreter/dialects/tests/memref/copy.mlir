// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @subview() -> memref<4x4xi32, strided<[4, 1], offset: 0>> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = memref.alloc() : memref<4x4xi32, strided<[4, 1], offset: 0>>
  %1 = memref.subview %0[%c1, %c1][%c2, %c2][%c1, %c1]
    : memref<4x4xi32, strided<[4, 1], offset: 0>> to
      memref<?x?xi32, strided<[?, ?], offset: ?>>

  %cst = arith.constant dense<[[1, 2], [3, 4]]> : memref<2x2xi32>
  memref.copy %cst, %1 : memref<2x2xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>

  return %0 : memref<4x4xi32, strided<[4, 1], offset: 0>>
}

// CHECK-LABEL: @subview
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]

func.func @strided() -> memref<4x4xi32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = memref.alloc() : memref<4x4xi32>
  %1 = memref.subview %0[%c1, %c1][%c2, %c2][%c2, %c2]
    : memref<4x4xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>

  %cst = arith.constant dense<[[1, 2], [3, 4]]> : memref<2x2xi32>
  memref.copy %cst, %1
    : memref<2x2xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>

  return %0 : memref<4x4xi32>
}

// CHECK-LABEL: @strided
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]
