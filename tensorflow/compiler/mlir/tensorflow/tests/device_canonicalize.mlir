// RUN: tf-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// Test empty launch with no results is folded away.
// CHECK-LABEL: func @empty_launch_no_results
func @empty_launch_no_results() {
  "tf_device.launch"() ( {
    tf_device.return
  }) {device = "device"} : () -> ()
  return
}

// CHECK-NOT: tf_device.launch


// Test empty launch with some results is folded away.
// CHECK-LABEL: func @empty_launch
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>)
func @empty_launch(%arg0 : tensor<i1>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i1>) {
  %result:2 = "tf_device.launch"() ( {
    tf_device.return %arg0, %arg1 : tensor<i1>, tensor<i32>
  }) {device = "device"} : () -> (tensor<i1>, tensor<i32>)
  return %result#1, %result#0 : tensor<i32>, tensor<i1>
}

// CHECK-NOT: tf_device.launch
// CHECK: return %[[ARG_1]], %[[ARG_0]]
