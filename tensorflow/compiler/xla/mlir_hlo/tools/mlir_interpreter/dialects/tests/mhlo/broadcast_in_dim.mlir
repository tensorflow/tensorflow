// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @broadcast() -> tensor<2x3xui16> {
  %0 = mhlo.constant dense<[1, 2]> : tensor<2xui16>
  %1 = "mhlo.broadcast_in_dim"(%0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<2xui16>) -> tensor<2x3xui16>
  return %1 : tensor<2x3xui16>
}

// CHECK{LITERAL}: [[1, 1, 1], [2, 2, 2]]

func.func @zero_rank() -> tensor<1x2x3xi32> {
  %in = mhlo.constant dense<1> : tensor<i32>
  %0 = "mhlo.broadcast_in_dim"(%in) {
    broadcast_dimensions = dense<[]> : tensor<0xi64>
  } : (tensor<i32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// CHECK{LITERAL}: [[[1, 1, 1], [1, 1, 1]]]
