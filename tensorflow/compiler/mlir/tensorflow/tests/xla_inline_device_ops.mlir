// RUN: tf-opt %s -tf-xla-inline-device-ops | FileCheck %s

// CHECK-LABEL: func @simple_stateful_partitioned_call
// CHECK-NOT: "tf_device.cluster"
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-NEXT: "tf.Const"
func.func @simple_stateful_partitioned_call(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ({
    %2 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {cluster_attr = "cluster_attr"} : () -> tensor<i32>
  %cst = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @stateful_partitioned_call_multiple_ops
// CHECK-NOT: "tf_device.cluster"
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-NOT: "tf_device.return"
// CHECK-NEXT: "tf.Const"
func.func @stateful_partitioned_call_multiple_ops(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ({
    %2 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> tensor<i32>
    %3 = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
    %4 = "tf.Add"(%2, %3) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
    tf_device.return %4 : tensor<i32>
  }) {cluster_attr = "cluster_attr"} : () -> tensor<i32>
  %5 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %6 = "tf.Add"(%0, %5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %6 : tensor<i32>
}

// -----

// CHECK-LABEL: func @no_stateful_partitioned_call_in_cluster_op
// CHECK-NOT: "tf_device.cluster"
// CHECK-NOT: "tf.StatefulPartitionedCall"
// CHECK: "tf.Const"
func.func @no_stateful_partitioned_call_in_cluster_op(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }) {cluster_attr = "cluster_attr"} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Add"(%2, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %3 : tensor<i32>
}

// -----

// CHECK-LABEL: func @multi_return_values_in_cluster_op
// CHECK "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK "tf.Add"(%cst_1, %cst_0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
// CHECK "tf.Add"(%cst_2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
func.func @multi_return_values_in_cluster_op(%arg0: tensor<i32>) -> () {
  %0, %1 = "tf_device.cluster"() ({
    %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
    tf_device.return %2, %3 : tensor<i32>, tensor<f32>
  }) {cluster_attr = "cluster_attr"} : () -> (tensor<i32>, tensor<f32>)
  %4 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
  %5 = "tf.Add"(%4, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %6 = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  %7 = "tf.Add"(%6, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return
}
