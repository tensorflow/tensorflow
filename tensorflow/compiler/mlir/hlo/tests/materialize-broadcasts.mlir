// RUN: mlir-hlo-opt -mhlo-test-materialize-broadcasts -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @clampBroadcast
// CHECK-SAME: (%[[MIN:.+]]: tensor<f32>, %[[VAL:.+]]: tensor<4xf32>, %[[MAX:.+]]: tensor<f32>)
func @clampBroadcast(%min: tensor<f32>, %value: tensor<4xf32>, %max: tensor<f32>) -> tensor<4xf32> {
  // CHECK-DAG: %[[MIN_BC:.+]] = "mhlo.broadcast"(%[[MIN]]) {broadcast_sizes = dense<4> : tensor<1xi64>} : (tensor<f32>) -> tensor<4xf32>
  // CHECK-DAG: %[[MAX_BC:.+]] = "mhlo.broadcast"(%[[MAX]]) {broadcast_sizes = dense<4> : tensor<1xi64>} : (tensor<f32>) -> tensor<4xf32>
  // CHECK: "mhlo.clamp"(%[[MIN_BC]], %[[VAL]], %[[MAX_BC]]) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %0 = "mhlo.clamp"(%min, %value, %max) : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
