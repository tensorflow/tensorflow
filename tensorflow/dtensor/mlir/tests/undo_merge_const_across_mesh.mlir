// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
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


