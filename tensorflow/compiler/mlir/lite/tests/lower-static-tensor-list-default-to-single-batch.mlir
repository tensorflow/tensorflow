// RUN: tf-opt "-tfl-lower-static-tensor-list=allow-tensorlist-pass-through default-to-single-batch=false" -split-input-file %s | FileCheck %s

// -----

func @tensorlistReserveConstantUnknownElementShapeDim(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<?x7xf32> {
  %cst = arith.constant dense<[-1, 7]> : tensor<2xi32>
  %0 = "tf.TensorListReserve"(%cst, %arg0) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x7xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg1, %cst) : (tensor<!tf_type.variant<tensor<?x7xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<?x7xf32>
  return %1 : tensor<?x7xf32>

// CHECK-LABEL: tensorlistReserveConstantUnknownElementShapeDim
// CHECK:      "tf.TensorListReserve"
// CHECK:      "tf.TensorListGetItem"
}
