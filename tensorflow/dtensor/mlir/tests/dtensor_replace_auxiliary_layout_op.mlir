// RUN: dtensor-opt %s -split-input-file -dtensor-replace-auxiliary-layout-op | FileCheck %s

// Check auxiliary `tf.DTensorLayout` is replaced by `tf.Identity`.
// CHECK-LABEL: func @check_replace_auxiliary_layout_op
func.func @check_replace_auxiliary_layout_op(%arg0: tensor<8x8xi32>) -> tensor<8x8xi32> {
  // CHECK-NEXT:  "tf.Identity"
  // CHECK-NEXT:  "tf.DTensorLayout"
  // CHECK-NEXT:  return
  %0 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}