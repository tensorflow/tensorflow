// RUN: tf-opt %s -constant-op-device-assignment | FileCheck %s

// CHECK: func @replace_const_op_test
func.func @replace_const_op_test() {
  // CHECK-NEXT: %[[RESULT_0:.*]] = "tf.Const"() {device = "/job:worker/replica:0/task:0/device:CPU:1", value = dense<2.000000e+00> : tensor<f32>}
  // CHECK-NEXT: %[[RESULT_1:.*]] = "tf.Const"() {device = "/job:worker/replica:0/task:0/device:CPU:0", value = dense<2.000000e+00> : tensor<f32>}
  // CHECK-NEXT: %[[RESULT_2:.*]] = "tf.AddV2"(%[[RESULT_1]], %[[RESULT_1]]) {device = "/job:worker/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_3:.*]] = "tf.AddV2"(%[[RESULT_0]], %[[RESULT_0]]) {device = "/job:worker/replica:0/task:0/device:CPU:1"}
  %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.AddV2"(%0, %0) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %2 = "tf.AddV2"(%0, %0) {device = "/job:worker/replica:0/task:0/device:CPU:1"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return
}

// CHECK: func @no_change_test
func.func @no_change_test() -> ()  {
  // CHECK-NEXT: %[[RESULT_0:.*]] = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  // CHECK-NEXT: %[[RESULT_1:.*]] = "tf.AddV2"(%[[RESULT_0]], %[[RESULT_0]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %1 = "tf.AddV2"(%0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  func.return
}

