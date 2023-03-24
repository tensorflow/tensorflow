// RUN: dtensor-opt %s -dtensor-handle_cross_cluster_dependences -split-input-file -verify-diagnostics | FileCheck %s

// Check that CopyToMesh op must be used to send tensors across mesh clusters.
func.func @main() {
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2 : tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)

    %2 = "tf_device.cluster"() ({
      // expected-error @+1 {{CopyToMeshOp must be used to send data across mesh}}
      %3 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
      tf_device.return %3 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check that CopyToMesh inside incorrect mesh cluster is disallowed.
func.func @main() -> tensor<i32> {
    // expected-error @+1 {{ Failed to extract mesh }}
    %0:2 = "tf_device.cluster"() ({
      %2 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2, %3 : tensor<i32>, tensor<i32>
    }) {_mesh="CPU|x=2,y=2|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>)


    %2 = "tf_device.cluster"() ({
      %3 = "tf.CopyToMesh"(%0#0) { layout ="sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : (tensor<i32>) -> (tensor<i32>)
      %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
    func.return %0#1 : tensor<i32>
}

// -----

// Check that Const op is cloned across mesh to reduce data transfer.
// CHECK-LABEL: func @main
func.func @main() -> tensor<i32> {
    // CHECK:        %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[CONST_OUT:.*]] = "tf.Const"()
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[CONST_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> tensor<i32>
    %0:2 = "tf_device.cluster"() ({
      %2 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2, %3 : tensor<i32>, tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[CONST_OUT:.*]] = "tf.Const"()
    // CHECK-NEXT:     %[[LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[CONST_OUT]])
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs: mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[LAYOUT_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %2 = "tf_device.cluster"() ({
      %3 = "tf.CopyToMesh"(%0#0) { layout ="sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : (tensor<i32>) -> (tensor<i32>)
      %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
    func.return %0#1 : tensor<i32>
}

// -----

// Check that CopyToMesh op is lowered to DTensorSend/DTensorRecv op.
// CHECK-LABEL: func @main
func.func @main() -> tensor<i32> {
    // CHECK:        %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.A"()
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[A_OUT]]
    // CHECK-NEXT:     "tf.DTensorSend"(%[[A_OUT]]
    // CHECK-SAME:     key = "communication_key_sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
    // CHECK-SAME:     target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> tensor<i32>
    %0:2 = "tf_device.cluster"() ({
      %2 = "tf.A"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2, %3 : tensor<i32>, tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[RECV_OUT:.*]] = "tf.DTensorRecv"()
    // CHECK-SAME:     key = "communication_key_sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[RECV_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %2 = "tf_device.cluster"() ({
      %3 = "tf.CopyToMesh"(%0#0) { layout ="sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : (tensor<i32>) -> (tensor<i32>)
      %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
    func.return %0#1 : tensor<i32>
}

// -----

// Check that tensor transfer from 2 clusters from same mesh without CopyToMesh is allowed.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:        %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[CONST_OUT:.*]] = "tf.Const"()
    // CHECK-NEXT:     %[[NEG_OUT_0:.*]] = "tf.Neg"(%[[CONST_OUT]]
    // CHECK-NEXT:     tf_device.return %[[NEG_OUT_0]]
    // CHECK-NEXT:   () -> tensor<i32>
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2 : tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     "tf.Const"()
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[NEG_OUT_2:.*]] = "tf.Neg"(%[[CLUSTER_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %5 = "tf_device.cluster"() ({
      %4 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check that CopyToMesh op with multiple usages is lowered to
// DTensorSend/DTensorRecv ops for each usages.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:        %[[CPU_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.A"()
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[A_OUT]]
    // CHECK-NEXT:     "tf.DTensorSend"(%[[NEG_OUT]]
    // CHECK-SAME:     key = "communication_key_sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
    // CHECK-SAME:     target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:     "tf.DTensorSend"(%[[NEG_OUT]]
    // CHECK-SAME:     key = "communication_key_sharding_specs:unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3_1"
    // CHECK-SAME:     target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> tensor<i32>
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2 : tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[RECV_OUT_1:.*]] = "tf.DTensorRecv"()
    // CHECK-SAME:     key = "communication_key_sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:     %[[NEG_OUT_1:.*]] = "tf.Neg"(%[[RECV_OUT_1]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %2 = "tf_device.cluster"() ({
      %3 = "tf.CopyToMesh"(%0) { layout ="sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : (tensor<i32>) -> (tensor<i32>)
      %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[RECV_OUT_2:.*]] = "tf.DTensorRecv"()
    // CHECK-SAME:     key = "communication_key_sharding_specs:unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3_1"
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>
    // CHECK-NEXT:     %[[NEG_OUT_2:.*]] = "tf.Neg"(%[[RECV_OUT_2]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %3 = "tf_device.cluster"() ({
      %4 = "tf.CopyToMesh"(%0) { layout ="sharding_specs:unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : (tensor<i32>) -> (tensor<i32>)
      %5 = "tf.Neg"(%4) : (tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh="GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[NEG_OUT_3:.*]] = "tf.Neg"(%[[CPU_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %4 = "tf_device.cluster"() ({
      %7 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
      tf_device.return %7 : tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)

    func.return
}

// -----

// Check that CopyToMesh inside cluster is replaced with Relayout.
// CHECK-LABEL: func @main
func.func @main() -> tensor<i32> {
    // CHECK:        %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[CONST_OUT:.*]] = "tf.Const"()
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[CONST_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> tensor<i32>
    %0:2 = "tf_device.cluster"() ({
      %2 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2, %3 : tensor<i32>, tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     %[[CONST_OUT:.*]] = "tf.Const"()
    // CHECK-NEXT:     %[[LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[CONST_OUT]])
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs: mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:     %[[RELAYOUT_OUT:.*]] = "tf.Relayout"(%[[LAYOUT_OUT]])
    // CHECK-SAME:     layout = "sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[RELAYOUT_OUT]]
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   () -> ()
    %2 = "tf_device.cluster"() ({
      %3 = "tf.CopyToMesh"(%0#0) { layout ="sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3" } : (tensor<i32>) -> (tensor<i32>)
      %4 = "tf.CopyToMesh"(%3) { layout ="sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3" } : (tensor<i32>) -> (tensor<i32>)
      %5 = "tf.Neg"(%4) : (tensor<i32>) -> tensor<i32>
      tf_device.return %5 : tensor<i32>
    }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
    func.return %0#1 : tensor<i32>
}

// -----

// Check that unused cluster results are removed.
// CHECK-LABEL: func @main
func.func @main() -> tensor<i32> {
    %0:3 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      tf_device.return %1, %2, %3 : tensor<i32>, tensor<i32>, tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>, tensor<i32>)
    func.return %0#2 : tensor<i32>
}

