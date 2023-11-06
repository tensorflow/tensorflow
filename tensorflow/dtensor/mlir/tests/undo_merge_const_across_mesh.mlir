// RUN: dtensor-opt %s -split-input-file -dtensor-undo-merge-const-across-mesh | FileCheck %s

// Check that constants with different meshes are duplicated.
// CHECK-LABEL: func @check_undo_sccp
func.func @check_undo_sccp() -> (tensor<4xi32>, tensor<4xi32>) {
    // CHECK-DAG: "tf.DTensorLayout"(%[[CONST_A:.*]]) <{global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>}> : (tensor<4xi32>) -> tensor<4xi32>
    // CHECK-DAG: %[[CONST_A]] = "tf.Const"()
    // CHECK-DAG: "tf.DTensorLayout"(%[[CONST_B:.*]]) <{global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>}> : (tensor<4xi32>) -> tensor<4xi32>
    // CHECK-DAG: %[[CONST_B]] = "tf.Const"()

    %cst = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4xi32>) -> tensor<4xi32>
    %3 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4xi32>) -> tensor<4xi32>
    func.return %2, %3 : tensor<4xi32>, tensor<4xi32>
}


