// Test DeviceIndex selector.

// RUN: tf-opt --tf-device-index-selector %s | FileCheck %s

// CHECK-LABEL: func @select
func @select(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  // CHECK:  %[[first:.*]] = "tf.DeviceIndex"
  // CHECK: constant dense<2>
  // CHECK:  return %[[first]],
  %0 = "tf.DeviceIndex"() {device = "", device_names = ["CPU", "GPU"]} : () -> tensor<i32>
  %1 = "tf.DeviceIndex"() {device = "", device_names = ["CPU", "GPU"]} : () -> tensor<i32>
  %4 = "tf.Case"(%1, %arg0, %arg1) {branches = [@sub, @add], output_shapes = [#tf_type.shape<>], is_stateless = false} : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>

  return %0, %4 : tensor<i32>, tensor<f32>
}

func @add(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sub(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
