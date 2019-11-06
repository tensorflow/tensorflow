// RUN: tf-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s --dump-input-on-failure

func @dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: "xla_hlo.dynamic-slice"
  %0 = xla_hlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = "xla_hlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<2xi64>) -> tensor<1x4xi32>
  return %1 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start
func @dynamic_slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[RESULT:.*]] =  "xla_hlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<3> : tensor<1xi64>
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>}
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  %0 = xla_hlo.constant dense<1> : tensor<1xi64>
  %2 = "xla_hlo.dynamic-slice"(%arg0, %0) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<1xi64>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_dynamic_shape
func @dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[RESULT:.*]] = "xla_hlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %0 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %1 = "xla_hlo.dynamic-slice"(%arg0, %0) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xi32>, tensor<2xi64>) -> tensor<1x4xi32>
  return %1 : tensor<1x4xi32>
}
