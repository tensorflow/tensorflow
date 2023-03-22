// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @collapse_shape()
    -> (tensor<1x6xi32>, tensor<2x3xi32>, tensor<6xi32>) {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %collapse1 = tensor.collapse_shape %cst [[0], [1, 2]]
      : tensor<1x2x3xi32> into tensor<1x6xi32>
  %collapse2 = tensor.collapse_shape %cst [[0, 1], [2]]
      : tensor<1x2x3xi32> into tensor<2x3xi32>
  %collapse3 = tensor.collapse_shape %cst [[0, 1, 2]]
      : tensor<1x2x3xi32> into tensor<6xi32>
  return %collapse1, %collapse2, %collapse3
      : tensor<1x6xi32>, tensor<2x3xi32>, tensor<6xi32>
}

// CHECK-LABEL: @collapse_shape
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 3, 4, 5, 6]]
// CHECK-NEXT{LITERAL}: [[1, 2, 3], [4, 5, 6]]
// CHECK-NEXT{LITERAL}: [1, 2, 3, 4, 5, 6]

func.func @to_unit() -> tensor<i32> {
  %cst = arith.constant dense<42> : tensor<1x1x1x1xi32>
  %collapse = tensor.collapse_shape %cst []
    : tensor<1x1x1x1xi32> into tensor<i32>
  return %collapse : tensor<i32>
}

// CHECK-LABEL: @to_unit
// CHECK-NEXT: Results
// CHECK-NEXT: 42
