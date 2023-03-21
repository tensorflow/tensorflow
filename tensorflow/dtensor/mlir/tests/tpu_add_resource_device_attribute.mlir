// RUN: dtensor-opt %s -split-input-file -dtensor-tpu-add-resource-device-attribute | FileCheck %s

// Test that tf.ReadVariable op and tf.AssignVariable op has device attribute
// added that is consistent with device attribute of TPUExecute op.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<!tf_type.resource<tensor<300x128xf32>>>, %arg1: tensor<!tf_type.resource<tensor<300x128xf32>>>) -> (tensor<300x128xf32>) {
  %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {config = "mesh:TPU,x=2,y=2", config_proto = "", executor_type = "", f = @tpu_func} : (tensor<!tf_type.resource<tensor<300x128xf32>>>, tensor<!tf_type.resource<tensor<300x128xf32>>>) -> (tensor<300x128xf32>)
  func.return %0 :tensor<300x128xf32>
}

// CHECK-LABEL: func @tpu_func
// CHECK-SAME:  %arg0: tensor<!tf_type.resource<tensor<300x128xf32>>>
// CHECK-SAME:  tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"
// CHECK-SAME:  %arg1: tensor<!tf_type.resource<tensor<300x128xf32>>>
// CHECK-SAME:  tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"
func.func @tpu_func(%arg0: tensor<!tf_type.resource<tensor<300x128xf32>>>, %arg1: tensor<!tf_type.resource<tensor<300x128xf32>>>) -> tensor<300x128xf32> {
  %0 = "tf.ReadVariableOp"(%arg0) {_global_shape = [#tf_type.shape<300x128>], _layout = ["mesh:TPU,x=2,y=2 layout:x,unsharded,"]} : (tensor<!tf_type.resource<tensor<300x128xf32>>>) -> tensor<300x128xf32>

  %1:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() {
      metadata = "...",
      mlir_module = ".."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  "tf_device.launch"() ({
    "tf.TPUCompileSucceededAssert"(%1#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()

  %2 = "tf_device.launch"() ({
    %3 = "tf.TPUExecute"(%0, %1#1) : (tensor<300x128xf32>, tensor<2x!tf_type.string>) -> tensor<300x128xf32>
    tf_device.return %3 : tensor<300x128xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<300x128xf32>

  "tf.AssignVariableOp"(%arg1, %2) : (tensor<!tf_type.resource<tensor<300x128xf32>>>, tensor<300x128xf32>) -> ()
  func.return %2 : tensor<300x128xf32>
}

// -----

// Test that device attribute to resource input to TPU computation is correctly
// added.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<!tf_type.resource<tensor<300x128xf32>>>) -> (tensor<300x128xf32>) {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "mesh:TPU,x=2,y=2", config_proto = "", executor_type = "", f = @tpu_func} : (tensor<!tf_type.resource<tensor<300x128xf32>>>) -> (tensor<300x128xf32>)
  func.return %0 :tensor<300x128xf32>
}

// CHECK-LABEL: func @tpu_func
// CHECK-SAME:  %arg0: tensor<!tf_type.resource<tensor<300x128xf32>>>
// CHECK-SAME:  tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"
func.func @tpu_func(%arg0: tensor<!tf_type.resource<tensor<300x128xf32>>>) -> tensor<300x128xf32> {
  %1:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() {
      metadata = "...",
      mlir_module = ".."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  "tf_device.launch"() ({
    "tf.TPUCompileSucceededAssert"(%1#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()

  %2 = "tf_device.launch"() ({
    %3 = "tf.TPUExecute"(%arg0, %1#1) : (tensor<!tf_type.resource<tensor<300x128xf32>>>, tensor<2x!tf_type.string>) -> tensor<300x128xf32>
    tf_device.return %3 : tensor<300x128xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<300x128xf32>

  func.return %2 : tensor<300x128xf32>
}


