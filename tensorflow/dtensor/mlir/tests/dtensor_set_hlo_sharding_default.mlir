// RUN: dtensor-opt %s -split-input-file -dtensor-set-hlo-sharding | FileCheck %s

// Check all inputs and operations have sharding attributes, with `check_layout_use_xla_spmd` set to default value (false).
// CHECK-LABEL: func @check_layouts_are_converted_to_xla_sharding_attributes
// CHECK-SAME: (%arg0: tensor<8x8xi32> {mhlo.sharding = "", tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<8x8xi32> {mhlo.sharding = ""}) {
func.func @check_layouts_are_converted_to_xla_sharding_attributes(
  %arg0: tensor<8x8xi32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"}) -> tensor<8x8xi32> {
  // CHECK:      "tf.DTensorLayout"
  // CHECK:      "tf.Identity"
  // CHECK:      "tf.DTensorLayout"
  // CHECK-NEXT: return
  %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %2 = "tf.Identity"(%1) {_global_shape = [#tf_type.shape<8x8>], device = ""} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %3 : tensor<8x8xi32>
}
