// RUN: mlir-hlo-opt -mhlo-legalize-broadcast-to-broadcast-in-dim -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @broadcast_to_broadcast_in_dim
func.func @broadcast_to_broadcast_in_dim(%arg0: tensor<4xi64>) -> tensor<1x2x3x4xi64> {
  // CHECK: [[RES:%.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<4xi64>) -> tensor<1x2x3x4xi64>
  %0 = "mhlo.broadcast"(%arg0) {
    broadcast_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<4xi64>) -> tensor<1x2x3x4xi64>
  // CHECK: return [[RES]] : tensor<1x2x3x4xi64>
  func.return %0 : tensor<1x2x3x4xi64>
}

// -----

// CHECK-LABEL: @broadcast_to_broadcast_in_dim_dynamic_operand
func.func @broadcast_to_broadcast_in_dim_dynamic_operand(%arg0: tensor<?x4xi64>) -> tensor<1x2x3x4xi64> {
  // CHECK: [[RES:%.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>}> : (tensor<?x4xi64>) -> tensor<1x2x3x4xi64>
  %0 = "mhlo.broadcast"(%arg0) {
    broadcast_sizes = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x4xi64>) -> tensor<1x2x3x4xi64>
  func.return %0 : tensor<1x2x3x4xi64>
}

// -----

// CHECK-LABEL: @broadcast_to_broadcast_in_dim_dynamic_result
func.func @broadcast_to_broadcast_in_dim_dynamic_result(%arg0: tensor<3x4xi64>) -> tensor<1x2x?x4xi64> {
  // CHECK: "mhlo.broadcast"
  %0 = "mhlo.broadcast"(%arg0) {
    broadcast_sizes = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi64>) -> tensor<1x2x?x4xi64>
  func.return %0 : tensor<1x2x?x4xi64>
}
