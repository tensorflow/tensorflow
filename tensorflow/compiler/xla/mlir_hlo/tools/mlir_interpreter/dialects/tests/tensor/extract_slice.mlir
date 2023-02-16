// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @extract() -> tensor<1x2x1xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %ret = tensor.extract_slice %cst[0, 0, 0][1, 2, 1][1, 1, 1] : tensor<1x2x3xi32> to tensor<1x2x1xi32>
  return %ret : tensor<1x2x1xi32>
}

// CHECK-LABEL: @extract
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1], [4]]]

func.func @rank_reduction() -> tensor<2xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %ret = tensor.extract_slice %cst[0, 0, 0][1, 2, 1][1, 1, 1] : tensor<1x2x3xi32> to tensor<2xi32>
  return %ret : tensor<2xi32>
}

// CHECK-LABEL: @rank_reduction
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 4]
