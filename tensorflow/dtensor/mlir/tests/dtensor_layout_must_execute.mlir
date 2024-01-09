// RUN: dtensor-opt %s -split-input-file -sccp -canonicalize | FileCheck %s

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xi32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> tensor<2x4xi32> attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
  // COM: DTensorLayout Op for not used argument arg1 must not be removed
  // CHECK: = "tf.DTensorLayout"(%arg1)
  %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
  %cst = "tf.Const"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"], _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", device = "", value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>} : () -> tensor<2x4xi32>
  %1 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
  %2 = "tf.Relayout"(%1) {device = "", layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : (tensor<2x4xi32>) -> tensor<2x4xi32>
  %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
  return %3 : tensor<2x4xi32>
}
