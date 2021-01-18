// RUN: tf-opt --cluster-tf-ops-by-host %s | FileCheck %s


// CHECK: func @main(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, %[[ARG_1:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"})
// CHECK-NEXT:   %[[RESULT_0:.*]]:2 = tf_device.remote_run "/job:worker/replica:0/task:0" @_job_worker_replica_0_task_0(%[[ARG_0]])
// CHECK-NEXT:   %[[RESULT_1:.*]] = tf_device.remote_run "/job:worker/replica:0/task:1" @_job_worker_replica_0_task_1(%[[ARG_1]])
// CHECK-NEXT:   %[[RESULT_2:.*]] = "tf.Const"() {value = dense<16> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:   %[[RESULT_3:.*]] = "tf.AddV2"(%[[RESULT_2]], %[[RESULT_2]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT:   return %[[RESULT_0]]#0, %[[RESULT_0]]#1, %[[RESULT_1]] : tensor<i32>, tensor<i32>, tensor<i32>

// CHECK: func @_job_worker_replica_0_task_0(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}) -> (tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:1"}) attributes {host = "/job:worker/replica:0/task:0"}
// CHECK-NEXT:   %[[RESULT_0:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]]) {device = "/job:worker/replica:0/task:0/device:CPU:0"}
// CHECK-NEXT:   %[[RESULT_1:.*]] = "tf.Mul"(%[[RESULT_0]], %[[RESULT_0]]) {device = "/job:worker/replica:0/task:0/device:CPU:0"}
// CHECK-NEXT:   %[[RESULT_2:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]]) {device = "/job:worker/replica:0/task:0/device:CPU:1"}
// CHECK-NEXT:   return %[[RESULT_1]], %[[RESULT_2]] : tensor<i32>, tensor<i32>

// CHECK: func @_job_worker_replica_0_task_1(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"}) -> (tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"}) attributes {host = "/job:worker/replica:0/task:1"}
// CHECK-NEXT:   %[[RESULT_0:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-NEXT:   return %[[RESULT_0]] : tensor<i32>

func @main(%arg0: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, %arg1: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"}) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %0 = "tf.AddV2"(%arg0, %arg0) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Mul"(%0, %0) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%arg0, %arg0) {device = "/job:worker/replica:0/task:0/device:CPU:1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.AddV2"(%arg1, %arg1) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %4 = "tf.Const"() { value = dense<16> : tensor<4xi32> } : () -> tensor<4xi32>
  %5 = "tf.AddV2"(%4, %4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %1, %2, %3 : tensor<i32>, tensor<i32>, tensor<i32>
}
