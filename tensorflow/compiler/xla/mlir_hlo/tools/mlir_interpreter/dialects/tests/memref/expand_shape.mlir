// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @expand_shape()
    -> (memref<1x2x3xi32>, memref<1x2x3xi32>, memref<2x1x3xi32>,
        memref<1x2x3xi32>) {
  %cst1 = arith.constant dense<[[1, 2, 3, 4, 5, 6]]> : memref<1x6xi32>
  %cst2 = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : memref<2x3xi32>
  %cst3 = arith.constant dense<[1, 2, 3, 4, 5, 6]> : memref<6xi32>
  %expand1 = memref.expand_shape %cst1 [[0], [1, 2]]
      : memref<1x6xi32> into memref<1x2x3xi32>
  %expand2 = memref.expand_shape %cst2 [[0, 1], [2]]
      : memref<2x3xi32> into memref<1x2x3xi32>
  %expand3 = memref.expand_shape %cst2 [[0, 1], [2]]
      : memref<2x3xi32> into memref<2x1x3xi32>
  %expand4 = memref.expand_shape %cst3 [[0, 1, 2]]
      : memref<6xi32> into memref<1x2x3xi32>
  return %expand1, %expand2, %expand3, %expand4
      : memref<1x2x3xi32>, memref<1x2x3xi32>, memref<2x1x3xi32>,
        memref<1x2x3xi32>
}

// CHECK-LABEL: @expand_shape
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 5, 6]]]
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 5, 6]]]
// CHECK-NEXT{LITERAL}: [[[1, 2, 3]], [[4, 5, 6]]]
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 5, 6]]]

func.func @zero_rank()
    -> (memref<1x1xi32>) {
  %cst = arith.constant dense<1> : memref<i32>
  %expand = memref.expand_shape %cst []
      : memref<i32> into memref<1x1xi32>
  return %expand : memref<1x1xi32>
}

// CHECK-LABEL: @zero_rank
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1]]
