// RUN: tf-tfrt-opt -optimize-tf-for-tfrt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @fold_device_index
func @fold_device_index() -> tensor<i32> {
  // CHECK-NOT: tf.DeviceIndex
  // CHECK: tf.Const
  // CHECK-SAME: value = dense<1> : tensor<i32>
  %0 = "tf.DeviceIndex"() {device = "/device:CPU:0", device_names = ["GPU", "CPU"]} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @not_fold_device_index
func @not_fold_device_index() -> tensor<i32> {
  // CHECK-NOT: tf.Const
  // CHECK: tf.DeviceIndex
  %0 = "tf.DeviceIndex"() {device = "", device_names = ["CPU", "GPU"]} : () -> tensor<i32>
  return %0 : tensor<i32>
}
