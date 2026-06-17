// RUN: dtensor-opt -- %s -split-input-file -dtensor-annotate-global-shape -dtensor-sparse-expansion -verify-diagnostics | FileCheck %s

// Check SparseExpansion for tf.MatMul with 1 SparseTensor left operand expands to SparseTensorDenseMatMul op.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<4x16xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}, %arg2: tensor<?x2xi64>, %arg3: tensor<2xi64>, %arg4: tensor<?xf32>) -> tensor<8x16xf32> {
  // CHECK: "tf.Const"
  // CHECK-NEXT: "tf.SparseToDense"
  // CHECK-NEXT: %[[DENSE:.*]] = "tf.DTensorLayout"(%arg1)
  // CHECK-NEXT: "tf.DTensorLayout"
  // CHECK-NEXT: "tf.SparseTensorDenseMatMul"(%arg2, %arg4, %arg3, %[[DENSE]])
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.SparseToDense"(%arg2, %arg3, %arg4, %cst) : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xf32>, tensor<f32>) -> tensor<8x4xf32>
  %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x16>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %2 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<8x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %3 = "tf.MatMul"(%2, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %5 = "tf.Identity"(%2) {device = ""} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  func.return %4 : tensor<8x16xf32>
}

// -----


// Check that after SparseExpansion, unused SparseToDense ops are removed.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<4x16xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}, %arg2: tensor<?x2xi64>, %arg3: tensor<2xi64>, %arg4: tensor<?xf32>) -> tensor<8x16xf32> {
  // CHECK: %[[DENSE:.*]] = "tf.DTensorLayout"(%arg1)
  // CHECK-NEXT: "tf.SparseTensorDenseMatMul"(%arg2, %arg4, %arg3, %[[DENSE]])
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.SparseToDense"(%arg2, %arg3, %arg4, %cst) : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xf32>, tensor<f32>) -> tensor<8x4xf32>
  %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x16>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %2 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<8x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %3 = "tf.MatMul"(%2, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %4 : tensor<8x16xf32>
}

// -----

// Check SparseExpansion for tf.MatMul with 2 SparseTensor operands is a no-change. That is, the original op gets returned.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>, %arg3: tensor<?xf32>, %arg4: tensor<?x2xi64>, %arg5: tensor<2xi64>, %arg6: tensor<?xf32>) -> tensor<8x16xf32> {
  // CHECK: "tf.Const"
  // CHECK-NEXT: "tf.SparseToDense"
  // CHECK: "tf.Const"
  // CHECK: "tf.SparseToDense"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.MatMul"
  // CHECK: "tf.Identity"
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.SparseToDense"(%arg4, %arg5, %arg6, %cst) : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xf32>, tensor<f32>) -> tensor<4x16xf32>
  %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.SparseToDense"(%arg1, %arg2, %arg3, %cst_0) : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xf32>, tensor<f32>) -> tensor<8x4xf32>
  %2 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<4x16>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|batch=2,x=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %3 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<8x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|batch=2,x=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %4 = "tf.MatMul"(%3, %2) {device = "", transpose_a = false, transpose_b = false} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %5 : tensor<8x16xf32>
}
