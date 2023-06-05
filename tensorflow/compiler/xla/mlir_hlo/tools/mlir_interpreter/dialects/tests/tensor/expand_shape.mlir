// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @expand_shape()
    -> (tensor<1x6xi32>, tensor<2x3xi32>, tensor<1x2x3xi32>) {
  %cst = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %expand1 = tensor.expand_shape %cst [[0, 1]]
      : tensor<6xi32> into tensor<1x6xi32>
  %expand2 = tensor.expand_shape %cst [[0, 1]]
      : tensor<6xi32> into tensor<2x3xi32>
  %expand3 = tensor.expand_shape %cst [[0, 1, 2]]
      : tensor<6xi32> into tensor<1x2x3xi32>
  return %expand1, %expand2, %expand3
      : tensor<1x6xi32>, tensor<2x3xi32>, tensor<1x2x3xi32>
}

// CHECK-LABEL: @expand_shape
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 3, 4, 5, 6]]
// CHECK-NEXT{LITERAL}: [[1, 2, 3], [4, 5, 6]]
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 5, 6]]]

func.func @from_unit() -> tensor<1x1xi32> {
  %cst = arith.constant dense<42> : tensor<i32>
  %expand = tensor.expand_shape %cst [] : tensor<i32> into tensor<1x1xi32>
  return %expand : tensor<1x1xi32>
}

// CHECK-LABEL: @from_unit
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<1x1xi32>: [[42]]

func.func @dynamic()
    -> (tensor<?x2x3xi32>) {
  %cst = arith.constant dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x6xi32>
  %cst_cast = tensor.cast %cst : tensor<1x6xi32> to tensor<?x6xi32>
  %expand1 = tensor.expand_shape %cst_cast [[0], [1, 2]]
      : tensor<?x6xi32> into tensor<?x2x3xi32>
  return %expand1 : tensor<?x2x3xi32>
}

// CHECK-LABEL: @dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 5, 6]]]

func.func @dynamic2() -> (tensor<?x3xi32>) {
  %cst = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %cst_cast = tensor.cast %cst : tensor<6xi32> to tensor<?xi32>
  %expand1 = tensor.expand_shape %cst_cast [[0, 1]]
      : tensor<?xi32> into tensor<?x3xi32>
  return %expand1 : tensor<?x3xi32>
}

// CHECK-LABEL: @dynamic2
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 3], [4, 5, 6]]
