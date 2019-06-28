// RUN: tf-opt %s -split-input-file | FileCheck %s

// Check unranked types
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<*xi32>) -> tensor<*xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<*xi32>):
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

// Check different input and output element types
func @broadcast_cmp_op(tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
  // CHECK: %0 = "tf.Less"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// Check unranked operand and result with different element types
func @broadcast_cmp_op(tensor<*xi32>, tensor<i32>) -> tensor<*xi1> {
^bb0(%arg0: tensor<*xi32>, %arg1: tensor<i32>):
  // CHECK: %0 = "tf.Less"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// -----

// Check a TF specific element type is accepted
func @broadcast_cmp_op(%arg0: tensor<4x2x!tf.string>, %arg1: tensor<2x!tf.string>) -> tensor<4x2x!tf.string> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x2x!tf.string>, tensor<2x!tf.string>) -> tensor<4x2x!tf.string>
  return %0 : tensor<4x2x!tf.string>
}
