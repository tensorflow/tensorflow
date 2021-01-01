// RUN: tf-opt --tf-cross-host-transfer %s | FileCheck %s

// CHECK-LABEL: func @test_merge_send
func @test_merge_send() {
  // CHECK-NEXT: %[[RESULT_0:.*]] = "tf.Const"() {device = "/job:worker/replica:0/task:0/device:CPU:0", value = dense<3.000000e+00> : tensor<f32>}
  %0 = "tf.Const"() {device = "/job:worker/replica:0/task:0/device:CPU:0", value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>

  // CHECK-NEXT: tf_device.send %[[RESULT_0]] "key-0" "/job:worker/replica:0/task:1" {device = "/job:worker/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_1:.*]] = tf_device.receive "key-0" "/job:worker/replica:0/task:0" {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_2:.*]] = "tf.Sqrt"(%[[RESULT_1]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  %1 = "tf.Sqrt"(%0) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: %[[RESULT_3:.*]] = "tf.Sqrt"(%[[RESULT_1]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  %2 = "tf.Sqrt"(%0) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<f32>) -> tensor<f32>
  return
}

// CHECK-LABEL: func @test_multiple_send
func @test_multiple_send() -> tensor<f32> {
  // CHECK-NEXT: %[[RESULT_0:.*]] = "tf.Const"() {device = "/job:worker/replica:0/task:0/device:CPU:0", value = dense<3.000000e+00> : tensor<f32>}
  %0 = "tf.Const"() {device = "/job:worker/replica:0/task:0/device:CPU:0", value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>

  // CHECK-NEXT: tf_device.send %[[RESULT_0]] "key-1" "/job:worker/replica:0/task:1" {device = "/job:worker/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_1:.*]] = tf_device.receive "key-1" "/job:worker/replica:0/task:0" {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_2:.*]] = "tf.Sqrt"(%[[RESULT_1]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  %1 = "tf.Sqrt"(%0) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: tf_device.send %[[RESULT_2]] "key-2" "/job:localhost/replica:0/task:0" {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_3:.*]] = tf_device.receive "key-2" "/job:worker/replica:0/task:1" {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_4:.*]] = "tf.Identity"(%[[RESULT_3]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  %2 = "tf.Identity"(%1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: return %[[RESULT_4]] : tensor<f32>
  return %2 : tensor<f32>
}

// CHECK: func @test_send_func_arg(%[[ARG_0:.*]]: tensor<f32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}) {
func @test_send_func_arg(%arg0: tensor<f32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}) {
  // CHECK-NEXT: tf_device.send %[[ARG_0]] "key-3" "/job:localhost/replica:0/task:0" {device = "/job:worker/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_0:.*]] = tf_device.receive "key-3" "/job:worker/replica:0/task:0" {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_1:.*]] = "tf.Identity"(%[[RESULT_0]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  %0 = "tf.Identity"(%arg0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<f32>) -> tensor<f32>

  return
}

// CHECK: func @test_not_send_while_loop_arg(%[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<*xf32>, %[[ARG_2:.*]]: tensor<i32>) {
func @test_not_send_while_loop_arg(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: tensor<i32>) {
  // CHECK-NEXT: %[[RESULT_0:.*]]:2 = "tf.WhileRegion"(%[[ARG_0]], %[[ARG_1]]) ( {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ( {
  // CHECK-NEXT: bb0(%[[ARG_3:.*]]: tensor<i32>, %[[ARG_4:.*]]: tensor<*xf32>)
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<*xf32>):
    // CHECK-NEXT: %[[RESULT_1:.*]] = "tf.Identity"(%[[ARG_3]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"}
    %2 = "tf.Identity"(%arg3) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<i32>) -> tensor<i32>
    // CHECK-NEXT: tf_device.send %[[RESULT_1]] "key-4" "/job:localhost/replica:0/task:0" {device = "/job:worker/replica:0/task:1/device:CPU:0"}
    // CHECK-NEXT: %[[RESULT_2:.*]] = tf_device.receive "key-4" "/job:worker/replica:0/task:1" {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
    // CHECK-NEXT: %[[RESULT_3:.*]] = "tf.NotEqual"(%[[ARG_2]], %[[RESULT_2]])
    %3 = "tf.NotEqual"(%arg2, %2) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tf.Yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<*xf32>):
    %cst = constant dense<1> : tensor<i32>
    %1 = "tf.Sub"(%arg3, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.Yield"(%1, %arg4) : (tensor<i32>, tensor<*xf32>) -> ()
  }) {is_stateless = true} : (tensor<i32>, tensor<*xf32>) -> (tensor<i32>, tensor<*xf32>)
  return
}
