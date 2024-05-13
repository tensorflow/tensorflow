// RUN: dtensor-opt -- %s -split-input-file -dtensor-infer-shapes-for-restorev2-op -verify-diagnostics | FileCheck %s

// Check the tf.RestoreV2Op's and all connected ops' resulting types are inferred from the AssignVariableOps in a single mesh. All unknown shapes should be known after this pass.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.string>, %arg2: tensor<2x!tf_type.string>, %arg3: tensor<2x!tf_type.string>, %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>>, %arg5: tensor<*x!tf_type.resource<tensor<i64>>>) {
    // CHECK:        %0:2 = "tf.RestoreV2"(%arg1, %arg2, %arg3) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<4x8xf32>, tensor<i64>)
    // CHECK-NEXT:   "tf.AssignVariableOp"(%arg4, %0#0) <{validate_shape = true}> : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
    // CHECK:        %1 = "tf.Identity"(%0#1) : (tensor<i64>) -> tensor<i64>
    // CHECK-NEXT:   "tf.AssignVariableOp"(%arg5, %1) <{validate_shape = false}> : (tensor<*x!tf_type.resource<tensor<i64>>>, tensor<i64>) -> ()
    %0:2 = "tf.RestoreV2"(%arg1, %arg2, %arg3): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<*xf32>, tensor<*xi64>)
    "tf.AssignVariableOp"(%arg4, %0#0) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<*xf32>) -> ()
    %1 = "tf.Identity"(%0#1) {} : (tensor<*xi64>) -> tensor<*xi64>
    "tf.AssignVariableOp"(%arg5, %1) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<i64>>>, tensor<*xi64>) -> ()
    func.return
}

// -----


// Check the tf.RestoreV2Op's and all connected ops' resulting types are inferred from the AssignVariableOps in cross mesh cluster. All unknown shapes should be known after this pass.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.string>, %arg2: tensor<2x!tf_type.string>, %arg3: tensor<2x!tf_type.string>, %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>>, %arg5: tensor<*x!tf_type.resource<tensor<i64>>>) {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:       %2 = "tf.DTensorRecv"() <{key = "communication_key_|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3", mesh = #dtensor.mesh<|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>, shape = #tf_type.shape<4x8>}> : () -> tensor<4x8xf32>
    // CHECK-NEXT:       %3 = "tf.Identity"(%2) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:       "tf.AssignVariableOp"(%arg4, %3) <{validate_shape = true}> : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
    // CHECK-NEXT:       tf_device.return
    "tf_device.cluster"() ({
      %1 = "tf.DTensorRecv"() {key = "communication_key_|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3", mesh = #dtensor.mesh<|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>, shape = #tf_type.shape<*>} : () -> tensor<*xf32>
      %2 = "tf.Identity"(%1) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.AssignVariableOp"(%arg4, %2) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<*xf32>) -> ()
      tf_device.return
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>, tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:       %2:2 = "tf.RestoreV2"(%arg1, %arg2, %arg3) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<4x8xf32>, tensor<i64>)
    // CHECK-NEXT:       %3 = "tf.Identity"(%2#1) : (tensor<i64>) -> tensor<i64>
    // CHECK-NEXT:       "tf.AssignVariableOp"(%arg5, %3) <{validate_shape = false}> : (tensor<*x!tf_type.resource<tensor<i64>>>, tensor<i64>) -> ()
    // CHECK-NEXT:       "tf.DTensorSend"(%2#0) <{key = "communication_key_|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3", target_mesh = #dtensor.mesh<|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>}> : (tensor<4x8xf32>) -> ()
    // CHECK-NEXT:       tf_device.return
    "tf_device.cluster"() ({
      %6:2 = "tf.RestoreV2"(%arg1, %arg2, %arg3) {} : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<*xf32>, tensor<*xi64>)
      %7 = "tf.Identity"(%6#1) : (tensor<*xi64>) -> tensor<*xi64>
      "tf.AssignVariableOp"(%arg5, %7) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<i64>>>, tensor<*xi64>) -> ()
      "tf.DTensorSend"(%6#0) {key = "communication_key_|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3", target_mesh = #dtensor.mesh<|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>} : (tensor<*xf32>) -> ()
      tf_device.return
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check correctness of shape inference and element type propagation of a graph containing tf.Cast ops.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.string>, %arg2: tensor<2x!tf_type.string>, %arg3: tensor<2x!tf_type.string>, %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>>, %arg5: tensor<*x!tf_type.resource<tensor<f32>>>) {
    // CHECK:        %0:2 = "tf.RestoreV2"(%arg1, %arg2, %arg3) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<4x8xf32>, tensor<bf16>)
    // CHECK-NEXT:   "tf.AssignVariableOp"(%arg4, %0#0) <{validate_shape = true}> : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
    // CHECK:        %1 = "tf.Cast"(%0#1) <{Truncate = false}> : (tensor<bf16>) -> tensor<f32>
    // CHECK-NEXT:   "tf.AssignVariableOp"(%arg5, %1) <{validate_shape = false}> : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %0:2 = "tf.RestoreV2"(%arg1, %arg2, %arg3): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<*xf32>, tensor<*xbf16>)
    "tf.AssignVariableOp"(%arg4, %0#0) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<*xf32>) -> ()
    %1 = "tf.Cast"(%0#1) {} : (tensor<*xbf16>) -> tensor<*xf32>
    "tf.AssignVariableOp"(%arg5, %1) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*xf32>) -> ()
    func.return
}
