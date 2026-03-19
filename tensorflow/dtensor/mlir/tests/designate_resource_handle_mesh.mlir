// RUN: dtensor-opt %s -split-input-file -dtensor-designate-resource-handle-mesh  -verify-diagnostics | FileCheck %s

// Check that pass is no-op for tf_device.cluster ops that does not contain
// tf.VarHandle / tf.DestroyResource op.
// CHECK-LABEL: func @main
func.func @main()  -> (tensor<i32>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.A"
  // CHECK-NEXT:   tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() : () -> tensor<i32>
    tf_device.return %1 : tensor<i32>
  }) {_mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<i32>)
  func.return %0 : tensor<i32>
}
 
// -----

// Check that empty mesh is assigned to cluster containing tf.VarHandle op.
// CHECK-LABEL: func @check_empty_mesh_assigned_varhandle_op
func.func @check_empty_mesh_assigned_varhandle_op()  -> (tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   %[[RESOURCE_OUT:.*]] = "tf.VarHandleOp"()
  // CHECK-NEXT:   tf_device.return %[[RESOURCE_OUT]]
  // CHECK-NEXT: _mesh = "empty_mesh"
  %1 = "tf_device.cluster"() ({
    %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
    tf_device.return %0 : tensor<!tf_type.resource<tensor<i32>>>
  }) : () -> (tensor<!tf_type.resource<tensor<i32>>>)
  func.return %1 : tensor<!tf_type.resource<tensor<i32>>>
}

// -----

// Check that non-empty mesh is assigned to cluster containing tf.DestroyResource op.
// CHECK-LABEL: func @check_mesh_assigned_destroy_resource_op
func.func @check_mesh_assigned_destroy_resource_op(%arg0: tensor<!tf_type.resource>)  -> () {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.DestroyResourceOp"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
  "tf_device.cluster"() ({
    "tf.DestroyResourceOp"(%arg0) : (tensor<!tf_type.resource>) -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> ()
  func.return
}

