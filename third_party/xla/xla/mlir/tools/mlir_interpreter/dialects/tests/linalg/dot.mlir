// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dot() -> tensor<i32> {
  %lhs = arith.constant dense<[1, 2]> : tensor<2xi32>
  %rhs = arith.constant dense<[3, 4]> : tensor<2xi32>
  %init = tensor.empty() : tensor<i32>
  %ret = linalg.dot ins(%lhs, %rhs: tensor<2xi32>, tensor<2xi32>)
                    outs(%init: tensor<i32>) -> tensor<i32>
  return %ret : tensor<i32>
}

// CHECK-LABEL: @dot
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: 11
