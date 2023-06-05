// RUN: dtensor-opt %s -split-input-file -dtensor-replace-relayout-with-identity | FileCheck %s

module {
  // CHECK-LABEL: main
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<2x4xi32> {tf._global_shape = #tf_type.shape<2x4>, tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<2x4xi32> {tf._global_shape = #tf_type.shape<2x4>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    %0 = "tf_device.cluster"() ({
      %1 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<2x4>], global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
      // CHECK:      "tf.Identity"
      // CHECK-NOT:  "tf.Relayout"
      // CHECK-SAME: _global_shape = [#tf_type.shape<2x4>], device = "", layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"
      %2 = "tf.Relayout"(%1) {_global_shape = [#tf_type.shape<2x4>], device = "", layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : (tensor<2x4xi32>) -> tensor<2x4xi32>
      %3 = "tf.DTensorLayout"(%2) {_global_shape = [#tf_type.shape<2x4>], global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
      tf_device.return %3 : tensor<2x4xi32>
    }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : () -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }
}

