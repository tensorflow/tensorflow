// RUN: dtensor-opt %s -dtensor-sparse-tensor-to-dense-tensor -split-input-file -verify-diagnostics | FileCheck %s

// Check int32 SparseTensors expand to SparseToDenseOp.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<6x4xi32> {tf._layout = "sharding_specs:batch,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", tf._sparse = true}) -> tensor<6x4xi32> attributes {tf.entry_function = {inputs = "device_id,op_input_0"}}{
  // CHECK: func @main(%arg0: tensor<i32>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>, %arg3: tensor<?xi32>) -> tensor<6x4xi32> attributes {tf.entry_function = {inputs = "device_id,op_input_sparse_indices_1,op_input_sparse_dense_shapes_1,op_input_sparse_values_1"}} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DENSE:.*]] = "tf.SparseToDense"(%arg1, %arg2, %arg3, %[[CST]])
  // CHECK-NEXT: %[[DENSE_OUT:.*]] = "tf.DTensorLayout"(%[[DENSE]])
  // CHECK-NEXT: "tf.AddV2"(%[[DENSE_OUT:.*]], %[[DENSE_OUT:.*]])
  // CHECK-NEXT: "tf.Identity"
  %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2,x=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<6x4xi32>) -> tensor<6x4xi32>
  %1 = "tf.AddV2"(%0, %0) {} : (tensor<6x4xi32>, tensor<6x4xi32>) -> tensor<6x4xi32>
  %2 = "tf.Identity"(%1) {} : (tensor<6x4xi32>) -> tensor<6x4xi32>
  func.return %2 : tensor<6x4xi32>
}

// -----

// Check float32 SparseTensors expand to SparseToDenseOp.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<6x4xf32> {tf._layout = "sharding_specs:batch,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", tf._sparse = true}) -> tensor<6x4xf32> attributes {tf.entry_function = {inputs = "device_id,op_input_0"}}{
  // CHECK: func @main(%arg0: tensor<i32>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>, %arg3: tensor<?xf32>) -> tensor<6x4xf32> attributes {tf.entry_function = {inputs = "device_id,op_input_sparse_indices_1,op_input_sparse_dense_shapes_1,op_input_sparse_values_1"}} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DENSE:.*]] = "tf.SparseToDense"(%arg1, %arg2, %arg3, %[[CST]])
  // CHECK-NEXT: %[[DENSE_OUT:.*]] = "tf.DTensorLayout"(%[[DENSE]])
  // CHECK-NEXT: "tf.AddV2"(%[[DENSE_OUT:.*]], %[[DENSE_OUT:.*]])
  // CHECK-NEXT: "tf.Identity"
  %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2,x=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
  %1 = "tf.AddV2"(%0, %0) {} : (tensor<6x4xf32>, tensor<6x4xf32>) -> tensor<6x4xf32>
  %2 = "tf.Identity"(%1) {} : (tensor<6x4xf32>) -> tensor<6x4xf32>
  func.return %2 : tensor<6x4xf32>
}

// -----

// Check int64 SparseTensors expand to SparseToDenseOp.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<6x4xi64> {tf._layout = "sharding_specs:batch,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", tf._sparse = true}) -> tensor<6x4xi64> attributes {tf.entry_function = {inputs = "device_id,op_input_0"}}{
  // CHECK: func @main(%arg0: tensor<i32>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>, %arg3: tensor<?xi64>) -> tensor<6x4xi64> attributes {tf.entry_function = {inputs = "device_id,op_input_sparse_indices_1,op_input_sparse_dense_shapes_1,op_input_sparse_values_1"}} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DENSE:.*]] = "tf.SparseToDense"(%arg1, %arg2, %arg3, %[[CST]])
  // CHECK-NEXT: %[[DENSE_OUT:.*]] = "tf.DTensorLayout"(%[[DENSE]])
  // CHECK-NEXT: "tf.AddV2"(%[[DENSE_OUT:.*]], %[[DENSE_OUT:.*]])
  // CHECK-NEXT: "tf.Identity"
  %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2,x=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<6x4xi64>) -> tensor<6x4xi64>
  %1 = "tf.AddV2"(%0, %0) {device = ""} : (tensor<6x4xi64>, tensor<6x4xi64>) -> tensor<6x4xi64>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<6x4xi64>) -> tensor<6x4xi64>
  func.return %2 : tensor<6x4xi64>
}

// -----

// Check the SparseTensor components are appended to the end of the block argument list.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<6x4xi64> {tf._layout = "sharding_specs:batch,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", tf._sparse = true}, %arg3: tensor<i32>) -> tensor<6x4xi64> attributes {tf.entry_function = {inputs = "device_id,op_input_0"}}{
  // CHECK: func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<?x2xi64>, %arg3: tensor<2xi64>, %arg4: tensor<?xi64>) -> tensor<6x4xi64> attributes {tf.entry_function = {inputs = "device_id,op_input_sparse_indices_1,op_input_sparse_dense_shapes_1,op_input_sparse_values_1"}} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DENSE:.*]] = "tf.SparseToDense"(%arg2, %arg3, %arg4, %[[CST]])
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NEXT: %[[DENSE_OUT:.*]] = "tf.DTensorLayout"(%[[DENSE]])
  // CHECK-NEXT: "tf.AddV2"(%[[DENSE_OUT:.*]], %[[DENSE_OUT:.*]])
  %3 = "tf.Identity"(%arg3) {} : (tensor<i32>) -> tensor<i32>
  %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2,x=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<6x4xi64>) -> tensor<6x4xi64>
  %1 = "tf.AddV2"(%0, %0) {device = ""} : (tensor<6x4xi64>, tensor<6x4xi64>) -> tensor<6x4xi64>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<6x4xi64>) -> tensor<6x4xi64>
  func.return %2 : tensor<6x4xi64>
}

// -----

// Check that a single SparseToDenseOp is created for all usages of a single SparseTensor
func.func @main(%arg0: tensor<i32>, %arg1: tensor<6x4xi64> {tf._layout = "sharding_specs:batch,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", tf._sparse = true}) -> tensor<6x4xi64> attributes {tf.entry_function = {inputs = "device_id,op_input_0"}}{
  // CHECK: func @main(%arg0: tensor<i32>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>, %arg3: tensor<?xi64>) -> tensor<6x4xi64> attributes {tf.entry_function = {inputs = "device_id,op_input_sparse_indices_1,op_input_sparse_dense_shapes_1,op_input_sparse_values_1"}} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DENSE:.*]] = "tf.SparseToDense"(%arg1, %arg2, %arg3, %[[CST]])
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NEXT: "tf.AddV2"
  %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<6x4xi64>) -> tensor<6x4xi64>
  %2 = "tf.Identity"(%arg1) {device = ""} : (tensor<6x4xi64>) -> tensor<6x4xi64>
  %3 = "tf.AddV2"(%1, %2) {device = ""} : (tensor<6x4xi64>, tensor<6x4xi64>) -> tensor<6x4xi64>
  func.return %3 : tensor<6x4xi64>
}

