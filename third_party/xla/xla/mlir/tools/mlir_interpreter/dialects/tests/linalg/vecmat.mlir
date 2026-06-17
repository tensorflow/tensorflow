// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @vecmat() -> tensor<2xi32> {
  %lhs = arith.constant dense<[4, 5]> : tensor<2xi32>
  %rhs = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %init = tensor.empty() : tensor<2xi32>
  %ret = linalg.vecmat ins(%lhs, %rhs: tensor<2xi32>, tensor<2x2xi32>)
                       outs(%init: tensor<2xi32>) -> tensor<2xi32>
  return %ret : tensor<2xi32>
}

// CHECK-LABEL: @vecmat
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [19, 28]
