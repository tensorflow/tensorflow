// RUN: dtensor-opt -- %s -split-input-file -dtensor-infer-shapes-for-restorev2-op -dtensor-layout-propagation-v2 -verify-diagnostics | FileCheck %s

// Check the combination of inferring shape for restorev2 op and layout
// propagation. After running both passes, all unknown shapes from RestoreV2
// should be made known, and output layouts of RestoreV2 should match the
// resource tensors being assigned to.

// Single mesh
func.func @main(
  %arg0: tensor<i32>,
  %arg1: tensor<!tf_type.string>,
  %arg2: tensor<!tf_type.string>,
  %arg3: tensor<!tf_type.string>,
  %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>>) {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       %[[RESTORE:.*]] = "tf.RestoreV2"(%0, %1, %2) : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<4x8xf32>
    // CHECK-NEXT:       %[[DLAYOUT:.*]] = "tf.DTensorLayout"(%[[RESTORE]]) <{global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>}> : (tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:       "tf.AssignVariableOp"(%3, %[[DLAYOUT]]) <{validate_shape = true}> : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
    "tf_device.cluster"() ({
      %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %1 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %2 = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %3 = "tf.DTensorLayout"(%arg4) {global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>) -> tensor<*x!tf_type.resource<tensor<4x8xf32>>>
      %4 = "tf.RestoreV2"(%0, %1, %2): (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<*xf32>)
      "tf.AssignVariableOp"(%3, %4) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<*xf32>) -> ()
      tf_device.return
    }) {_mesh="CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
    func.return
}

// -----

// Single mesh with ops between the Restore and Assign.
func.func @main(
  %arg0: tensor<i32>,
  %arg1: tensor<!tf_type.string>,
  %arg2: tensor<!tf_type.string>,
  %arg3: tensor<!tf_type.string>,
  %arg4: tensor<*x!tf_type.resource<tensor<4x8xf64>>>) {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       "tf.DTensorLayout"
    // CHECK-NEXT:       %[[RESTORE:.*]]  = "tf.RestoreV2"(%0, %1, %2) : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<4x8xf32>
    // CHECK-NEXT:       %[[DLAYOUT:.*]]  = "tf.DTensorLayout"(%[[RESTORE]]) <{global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>}> : (tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:       %[[CAST:.*]]     = "tf.Cast"(%[[DLAYOUT]]) <{Truncate = false}> : (tensor<4x8xf32>) -> tensor<4x8xf64>
    // CHECK-NEXT:       %[[DLAYOUT2:.*]] = "tf.DTensorLayout"(%[[CAST]]) <{global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>}> : (tensor<4x8xf64>) -> tensor<4x8xf64>
    // CHECK-NEXT:       "tf.AssignVariableOp"(%3, %[[DLAYOUT2]]) <{validate_shape = true}> : (tensor<*x!tf_type.resource<tensor<4x8xf64>>>, tensor<4x8xf64>) -> ()
    "tf_device.cluster"() ({
      %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %1 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %2 = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %3 = "tf.DTensorLayout"(%arg4) {global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<*x!tf_type.resource<tensor<4x8xf64>>>) -> tensor<*x!tf_type.resource<tensor<4x8xf64>>>
      %4 = "tf.RestoreV2"(%0, %1, %2): (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<*xf32>)
      %5 = "tf.Cast"(%4) {} : (tensor<*xf32>) -> tensor<*xf64>
      "tf.AssignVariableOp"(%3, %5) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf64>>>, tensor<*xf64>) -> ()
      tf_device.return
    }) {_mesh="CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
    func.return
}

// -----

// Cross mesh with send/recv
func.func @main(
  %arg0: tensor<i32>,
  %arg1: tensor<!tf_type.string>,
  %arg2: tensor<!tf_type.string>,
  %arg3: tensor<!tf_type.string>,
  %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>>) {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:       %[[RESOURCE:.*]] = "tf.DTensorLayout"(%arg4)
    // CHECK-NEXT:       %[[RECV:.*]] = "tf.DTensorRecv"() <{
    // CHECK-SAME:       key = "communication_key_|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"
    // CHECK-SAME:       mesh = #dtensor.mesh<|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>
    // CHECK-SAME:       shape = #tf_type.shape<4x8>
    // CHECK-SAME:       source_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>
    // CHECK-SAME:       () -> tensor<4x8xf32>
    // CHECK-NEXT:       %[[RECV_DL:.*]] = "tf.DTensorLayout"(%[[RECV]])
    // CHECK-NEXT:       %[[IDENTITY:.*]] = "tf.Identity"(%[[RECV_DL]]) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:       %[[IDENTITY_DL:.*]] = "tf.DTensorLayout"(%[[IDENTITY]])
    // CHECK-NEXT:       "tf.AssignVariableOp"(%[[RESOURCE]], %[[IDENTITY_DL]]) <{validate_shape = true}> : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
    // CHECK-NEXT:       tf_device.return
    "tf_device.cluster"() ({
      %4 = "tf.DTensorLayout"(%arg4) {global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1>} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>) -> tensor<*x!tf_type.resource<tensor<4x8xf32>>>
      %5 = "tf.DTensorRecv"() {key = "communication_key_|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", mesh = #dtensor.mesh<|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>, shape = #tf_type.shape<*>} : () -> tensor<*xf32>
      %6 = "tf.Identity"(%5) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.AssignVariableOp"(%4, %6) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<*xf32>) -> ()
      tf_device.return
    }) {_mesh="TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<i32>, tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:       %[[DL1:.*]] = "tf.DTensorLayout"(%arg1)
    // CHECK-NEXT:       %[[DL2:.*]] = "tf.DTensorLayout"(%arg2)
    // CHECK-NEXT:       %[[DL3:.*]] = "tf.DTensorLayout"(%arg3)
    // CHECK-NEXT:       %[[RESTORE:.*]] = "tf.RestoreV2"(%[[DL1]], %[[DL2]], %[[DL3]]) : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<4x8xf32>
    // CHECK-NEXT:       "tf.DTensorLayout"(%[[RESTORE]]) <{global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>}> : (tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:       "tf.DTensorSend"
    // CHECK-NEXT:       tf_device.return
    "tf_device.cluster"() ({
      %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %1 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %2 = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %3 = "tf.RestoreV2"(%0, %1, %2) {} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<*xf32>)
      "tf.DTensorSend"(%3) {key = "communication_key_|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", target_mesh = #dtensor.mesh<|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<*xf32>) -> ()
      tf_device.return
    }) {_mesh="CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<i32>)
    func.return
}
