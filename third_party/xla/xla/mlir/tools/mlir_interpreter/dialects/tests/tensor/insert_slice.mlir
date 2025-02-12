// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @insert() -> tensor<1x3x3xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]> : tensor<1x3x3xi32>
  %cst_2 = arith.constant dense<[[[10], [11]]]> : tensor<1x2x1xi32>
  %ret = tensor.insert_slice %cst_2 into %cst[0, 1, 1][1, 2, 1][1, 1, 1]
    : tensor<1x2x1xi32> into tensor<1x3x3xi32>
  return %ret : tensor<1x3x3xi32>
}

// CHECK-LABEL: @insert
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 10, 6], [7, 11, 9]]]

func.func @rank_increase() -> tensor<1x3x3xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]> : tensor<1x3x3xi32>
  %cst_2 = arith.constant dense<[10, 11]> : tensor<2xi32>
  %ret = tensor.insert_slice %cst_2 into %cst[0, 1, 1][1, 2, 1][1, 1, 1]
    : tensor<2xi32> into tensor<1x3x3xi32>
  return %ret : tensor<1x3x3xi32>
}

// CHECK-LABEL: @rank_increase
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 10, 6], [7, 11, 9]]]
