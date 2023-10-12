// RUN: tf-opt %s -tf-standard-pipeline | FileCheck %s

// CHECK-LABEL: removeDeadReadVariableOp
func.func @removeDeadReadVariableOp(%arg0: tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32> {
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  func.return %0: tensor<f32>

 // CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
 // CHECK: return %0
}
