// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dot_2d() -> tensor<2x2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[[4, 5], [6, 7]]> : tensor<2x2xi32>
  %dot = "mhlo.dot"(%lhs, %rhs)
    : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %dot : tensor<2x2xi32>
}

// CHECK-LABEL: @dot_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[16, 19], [36, 43]]

func.func @dot_2d_1d() -> tensor<2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[4, 5]> : tensor<2xi32>
  %dot = "mhlo.dot"(%lhs, %rhs)
    : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %dot : tensor<2xi32>
}

// CHECK-LABEL: @dot_2d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [14, 32]

func.func @dot_1d_1d() -> tensor<i32> {
  %lhs = mhlo.constant dense<[1, 2]> : tensor<2xi32>
  %rhs = mhlo.constant dense<[4, 5]> : tensor<2xi32>
  %dot = "mhlo.dot"(%lhs, %rhs)
    : (tensor<2xi32>, tensor<2xi32>) -> tensor<i32>
  return %dot : tensor<i32>
}

// CHECK-LABEL: @dot_1d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: 14