// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @collapse_shape()
    -> (memref<1x2x3xi32>, memref<1x6xi32>, memref<2x3xi32>, memref<6xi32>) {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : memref<1x2x3xi32>
  %collapse1 = memref.collapse_shape %cst [[0], [1, 2]]
      : memref<1x2x3xi32> into memref<1x6xi32>
  %collapse2 = memref.collapse_shape %cst [[0, 1], [2]]
      : memref<1x2x3xi32> into memref<2x3xi32>
  %collapse3 = memref.collapse_shape %cst [[0, 1, 2]]
      : memref<1x2x3xi32> into memref<6xi32>
  return %cst, %collapse1, %collapse2, %collapse3
      : memref<1x2x3xi32>, memref<1x6xi32>, memref<2x3xi32>, memref<6xi32>
}

// CHECK-LABEL: @collapse_shape
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 5, 6]]]
// CHECK-NEXT{LITERAL}: [[1, 2, 3, 4, 5, 6]]
// CHECK-NEXT{LITERAL}: [[1, 2, 3], [4, 5, 6]]
// CHECK-NEXT{LITERAL}: [1, 2, 3, 4, 5, 6]

func.func @zero_dim()
    -> (memref<6x0xi32>) {
  %cst = arith.constant dense<> : memref<1x2x3x0xi32>
  %collapse = memref.collapse_shape %cst [[0, 1, 2], [3]]
      : memref<1x2x3x0xi32> into memref<6x0xi32>
  return %collapse : memref<6x0xi32>
}

// CHECK-LABEL: @zero_dim
// CHECK-NEXT: Results
// CHECK-NEXT: TensorOrMemref<6x0xi32>
