// RUN: tf-opt --cluster-tf-ops-by-host %s | FileCheck %s

// The @main function is a Multi-hosts function which contains two parts:
//   - A local subgraph which contains both local ops and remote_run kernel to
//     trigger remote subgraph
//   - A remote subgraph which contains remote ops on worker:1.
// CHECK: func @main(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %[[ARG_1:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"})
// CHECK-NEXT:   %[[RESULT_0:.*]] = "tf.While"(%[[ARG_0]])
// CHECK-SAME:   body = @while_body
// CHECK-SAME:   cond = @while_cond
// CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT:   %[[RESULT_1:.*]] = tf_device.remote_run "/job:worker/replica:0/task:1" @[[MAIN_PARTITION_0:.*]](%[[ARG_1]])
// CHECK-NEXT:   return %[[RESULT_0]], %[[RESULT_1]]
func @main(%arg0: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg1: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"}) -> (tensor<i32>, tensor<i32>) {
  %1 = "tf.While"(%arg0) {cond = @while_cond, body = @while_body, is_stateless = false, shape_invariant, device="/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>) -> (tensor<i32>)

  %2 = "tf.AddV2"(%arg1, %arg1) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1, %2 : tensor<i32>, tensor<i32>
}
// Subgraph of @main function that is placed on worker:1
// CHECK: func @[[MAIN_PARTITION_0]](%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"})
// CHECK-SAME:  host = "/job:worker/replica:0/task:1"
// CHECK-NEXT:   %[[RESULT_0:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]])
// CHECK-SAME:  device = "/job:worker/replica:0/task:1/device:CPU:0"
// CHECK-NEXT:   return %[[RESULT_0]]

// CHECK: func @while_cond(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"})
// CHECK-NEXT:   %[[RESULT_0:.*]] = "tf.Const"()
// CHECK-NEXT:   %[[RESULT_1:.*]] = "tf.Less"(%[[ARG_0]], %[[RESULT_0]])
// CHECK-SAME:  device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT:   return %[[RESULT_1]] : tensor<i1>
func @while_cond(%arg0: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

// The @while_body function is a Multi-hosts function which contains three
// parts:
//   - A local subgraph which contains both local ops and remote_run kernels to
//     trigger remote subgraphs
//   - Two remote subgraph which contains remote ops on worker:1 and worker:2.
// CHECK: func @while_body(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"})
// CHECK-NEXT:   %[[RESULT_0:.*]] = "tf.Const"()
// CHECK-NEXT:   %[[RESULT_1:.*]] = "tf.AddV2"(%[[ARG_0]], %[[RESULT_0]])
// CHECK-NEXT:   %[[RESULT_2:.*]] = "tf.Const"() {value = dense<16> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:   tf_device.send %[[RESULT_2]] "key-0" "/job:worker/replica:0/task:1/device:CPU:0"
// CHECK-SAME:  device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT:   tf_device.remote_run "/job:worker/replica:0/task:1" @[[BODY_PARTITION_0:.*]]() : () -> ()
// CHECK-NEXT:   tf_device.send %[[RESULT_2]]
// CHECK-NEXT:   tf_device.remote_run "/job:worker/replica:0/task:2" @[[BODY_PARTITION_1:.*]]() : () -> ()
// TODO(tf-runtime): Allow while body having remote inputs and outputs.
func @while_body(%arg0: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<i32>) {
  %0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.AddV2"(%arg0, %0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.Const"() { value = dense<16> : tensor<i32> } : () -> tensor<i32>
  tf_device.send %2 "key-0" "/job:worker/replica:0/task:1/device:CPU:0" {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : tensor<i32>
  %3 = tf_device.receive "key-0" "/job:localhost/replica:0/task:0/device:CPU:0" {device="/job:worker/replica:0/task:1/device:CPU:0"} : tensor<i32>
  %4 = "tf.AddV2"(%3, %3) {device = "/job:worker/replica:0/task:1/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>

  tf_device.send %2 "key-1" "/job:worker/replica:0/task:2/device:CPU:0" {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : tensor<i32>
  %5 = tf_device.receive "key-1" "/job:localhost/replica:0/task:0/device:CPU:0" {device="/job:worker/replica:0/task:2/device:CPU:0"} : tensor<i32>
  %6 = "tf.AddV2"(%5, %5) {device = "/job:worker/replica:0/task:2/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>

  return %1 : tensor<i32>
}

// Subgraph of @while_body function that is placed on worker:1
// CHECK: func @[[BODY_PARTITION_0]]() attributes {host = "/job:worker/replica:0/task:1"}
// CHECK-NEXT:   %[[RESULT_0:.*]] = tf_device.receive "key-0"
// CHECK-NEXT:   %[[RESULT_1:.*]] = "tf.AddV2"(%[[RESULT_0]], %[[RESULT_0]])
// CHECK-SAME:  device = "/job:worker/replica:0/task:1/device:CPU:0"

// Subgraph of @while_body function that is placed on worker:2
// CHECK: func @[[BODY_PARTITION_1]]() attributes {host = "/job:worker/replica:0/task:2"}
// CHECK-NEXT:   %[[RESULT_0:.*]] = tf_device.receive "key-1"
// CHECK-NEXT:   %[[RESULT_1:.*]] = "tf.AddV2"(%[[RESULT_0]], %[[RESULT_0]])
// CHECK-SAME:  device = "/job:worker/replica:0/task:2/device:CPU:0"
