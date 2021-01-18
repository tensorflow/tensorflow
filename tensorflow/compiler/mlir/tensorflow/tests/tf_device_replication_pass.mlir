// RUN: tf-opt --tf-device-replication %s | FileCheck %s

// CHECK: func @test_1(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, %[[ARG_1:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"})
func @test_1(%arg0: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, %arg1: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: %[[RESULT_0:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]]) {device = "/job:worker/replica:0/task:0/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_1:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]]) {device = "/job:worker/replica:0/task:0/device:CPU:1"}
  // CHECK-NEXT: %[[RESULT_2:.*]] = "tf.AddV2"(%[[ARG_1]], %[[ARG_1]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  // CHECK-NEXT: %[[RESULT_3:.*]] = "tf.AddV2"(%[[ARG_1]], %[[ARG_1]]) {device = "/job:worker/replica:0/task:1/device:CPU:1"}
  %0:4 = tf_device.replicate([%arg0, %arg0, %arg1, %arg1] as %arg2: tensor<i32>) {devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:CPU:1", "/job:worker/replica:0/task:1/device:CPU:0", "/job:worker/replica:0/task:1/device:CPU:1"]}, n = 4 : i32} {
    %1 = "tf.AddV2"(%arg2, %arg2) {device = "TPU_REPLICATED_CORE_0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }
  // CHECK-NEXT: return %[[RESULT_0]], %[[RESULT_1]], %[[RESULT_2]], %[[RESULT_3]]
  return %0#0, %0#1, %0#2, %0#3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}


// CHECK: func @test_2(%[[ARG_0:.*]]: tensor<i32>
func @test_2(%arg0: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %0:4 = tf_device.replicate() {n = 4 : i32} {
    tf_device.return %arg0 : tensor<i32>
  }
  // CHECK-NEXT: return %[[ARG_0]], %[[ARG_0]], %[[ARG_0]], %[[ARG_0]]
  return %0#0, %0#1, %0#2, %0#3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}
