// RUN: tf-opt %s -split-input-file -tf-xla-broadcast | FileCheck %s

// CHECK-LABEL: func @move_broadcast
func.func @move_broadcast(%arg0: tensor<f32>) -> () {
  // CHECK:      %[[ELEM:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK-NEXT: %[[SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi64>}> : () -> tensor<0xi64>
  // CHECK-NEXT: %[[FULL:.*]] = "tf.Fill"(%[[SHAPE]], %[[ELEM]]) : (tensor<0xi64>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: tf_device.replicate([%arg0, %[[FULL]]] as %[[REPVAR:.*]]: tensor<f32>) {n = 2 : i32} {
  // CHECK-NEXT:   %[[ID:.*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
  // CHECK-NEXT:     %[[IDINSIDE:.*]] = "tf.Identity"(%[[REPVAR]]) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     tf_device.return %[[IDINSIDE]] : tensor<f32>
  // CHECK-NEXT:   }) : () -> tensor<f32>
  // CHECK-NEXT:   "tf_device.cluster"() ({
  // CHECK-NEXT:     %[[GROUP:.*]] = "tf.Const"()
  // CHECK-SAME:       [0, 1]
  // CHECK-NEXT:     %[[REDUCED:.*]] = "tf.XlaAllReduce"(%[[ID]], %[[GROUP]]) <{mode = "CrossReplica", reduce_op = "Add"}> : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
  // CHECK-NEXT:     "tf.OpA"(%[[REDUCED]]) : (tensor<f32>) -> ()
  tf_device.replicate {n = 2 : i32} {
    "tf_device.cluster"() ({
      "tf.OpA"(%arg0) : (tensor<f32>) -> ()
      tf_device.return
    }) : () -> ()
  }
  func.return
}
