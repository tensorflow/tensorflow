// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-colocate-splits | FileCheck %s

// CHECK-LABEL: func @colocate_split_with_pred
func.func @colocate_split_with_pred() {
  // CHECK: Split
  // CHECK-SAME: _class = ["loc:@class"]
  tf_executor.graph {
    %c, %control0 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %a, %control1 = tf_executor.island wraps "tf.A"() {_class = ["loc:@class"]} : () -> (tensor<2xf32>)
    %s:2, %control2 = tf_executor.island wraps "tf.Split"(%c, %a) {num_split = 2 : i32} : (tensor<i32>, tensor<2xf32>) -> (tensor<1xf32>, tensor<1xf32>)
    tf_executor.fetch
  }
  func.return
}

// -----

// CHECK-LABEL: func @colocate_split_with_pred_results
func.func @colocate_split_with_pred_results() {
  // CHECK: Split
  // CHECK-SAME: _class = ["loc:@class"]
  tf_executor.graph {
    %c, %control0 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %a:2, %control1 = tf_executor.island wraps "tf.A"() {_class = ["loc:@class"]} : () -> (tensor<2xf32>, tensor<2xf32>)
    %s:2, %control2 = tf_executor.island wraps "tf.Split"(%c, %a#1) {num_split = 2 : i32} : (tensor<i32>, tensor<2xf32>) -> (tensor<1xf32>, tensor<1xf32>)
    tf_executor.fetch
  }
  func.return
}

// -----

// CHECK-LABEL: func @no_colocate_split_has_device
func.func @no_colocate_split_has_device() {
  // CHECK: Split
  // CHECK-NOT: _class = ["loc:@class"]
  tf_executor.graph {
    %c, %control0 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %a, %control1 = tf_executor.island wraps "tf.A"() {_class = ["loc:@class"]} : () -> tensor<2xf32>
    %s:2, %control2 = tf_executor.island wraps "tf.Split"(%c, %a) {num_split = 2 : i32, device = "device"} : (tensor<i32>, tensor<2xf32>) -> (tensor<1xf32>, tensor<1xf32>)
    tf_executor.fetch
  }
  func.return
}
