// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @subview() -> (memref<4x4xi32, strided<[4, 1], offset: 0>>,
                         memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = memref.alloc() : memref<4x4xi32, strided<[4, 1], offset: 0>>
  %1 = memref.subview %0[%c1, %c1][%c2, %c2][%c1, %c1]
    : memref<4x4xi32, strided<[4, 1], offset: 0>> to
      memref<?x?xi32, strided<[?, ?], offset: ?>>

  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  %i3 = arith.constant 3 : i32
  %i4 = arith.constant 4 : i32

  memref.store %i1, %1[%c0, %c0] : memref<?x?xi32, strided<[?, ?], offset: ?>>
  memref.store %i2, %1[%c0, %c1] : memref<?x?xi32, strided<[?, ?], offset: ?>>
  memref.store %i3, %1[%c1, %c0] : memref<?x?xi32, strided<[?, ?], offset: ?>>
  memref.store %i4, %1[%c1, %c1] : memref<?x?xi32, strided<[?, ?], offset: ?>>

  return %0, %1 : memref<4x4xi32, strided<[4, 1], offset: 0>>,
                  memref<?x?xi32, strided<[?, ?], offset: ?>>
}

// CHECK-LABEL: @subview
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[1, 2], [3, 4]]

func.func @strided() -> memref<?x?xi32, strided<[?, ?], offset: ?>> {
  %0 = arith.constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : memref<2x5xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %1 = memref.subview %0[%c0, %c1][%c2, %c2][%c1, %c2]
    : memref<2x5xi32> to
      memref<?x?xi32, strided<[?, ?], offset: ?>>

  return %1 : memref<?x?xi32, strided<[?, ?], offset: ?>>
}

// CHECK-LABEL: @strided
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[2, 4], [7, 9]]

func.func @subview_of_subview() -> (memref<?xi32, strided<[?], offset: ?>>,
                                     memref<?xi32, strided<[?], offset: ?>>) {
  %0 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : memref<10xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  %1 = memref.subview %0[%c1][%c4][%c2]
    : memref<10xi32> to memref<?xi32, strided<[?], offset: ?>>
  %2 = memref.subview %1[%c1][%c2][%c2]
    : memref<?xi32, strided<[?], offset: ?>> to
      memref<?xi32, strided<[?], offset: ?>>

  return %1, %2 : memref<?xi32, strided<[?], offset: ?>>,
                  memref<?xi32, strided<[?], offset: ?>>
}

// CHECK-LABEL: @subview_of_subview
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [2, 4, 6, 8]
// CHECK-NEXT{LITERAL}: [4, 8]

func.func @negative_stride() -> memref<?xi32, strided<[?], offset: ?>> {
  %0 = arith.constant dense<[1, 2, 3, 4]> : memref<4xi32>
  %c-1 = arith.constant -1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %1 = memref.subview %0[%c3][%c4][%c-1]
    : memref<4xi32> to memref<?xi32, strided<[?], offset: ?>>
  return %1 : memref<?xi32, strided<[?], offset: ?>>
}

// CHECK-LABEL: @negative_stride
// CHECK-NEXT: Results
// CHECK-NEXT: [4, 3, 2, 1]

func.func @negative_stride_of_subview() -> (memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>) {
  %0 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : memref<10xi32>
  %c-2 = arith.constant -2 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %1 = memref.subview %0[%c1][%c4][%c2]
    : memref<10xi32> to memref<?xi32, strided<[?], offset: ?>>
  %2 = memref.subview %1[%c3][%c2][%c-2]
    : memref<?xi32, strided<[?], offset: ?>> to
      memref<?xi32, strided<[?], offset: ?>>
  return %1, %2 : memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>
}

// CHECK-LABEL: @negative_stride_of_subview
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [2, 4, 6, 8]
// CHECK-NEXT{LITERAL}: [8, 4]

func.func @rank_reduce_middle() -> memref<1x2x3xi32> {
  %a = arith.constant dense<[[[[0, 1, 2], [3, 4, 5]],
                              [[6, 7, 8], [9, 10,11]]]]>
    : memref<1x2x2x3xi32>
  %b = memref.subview %a[0, 0, 0, 0][1, 2, 1, 3][1, 1, 1, 1]
    : memref<1x2x2x3xi32> to memref<1x2x3xi32>
  return %b : memref<1x2x3xi32>
}

// CHECK-LABEL: @rank_reduce_middle
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[0, 1, 2], [6, 7, 8]]]
