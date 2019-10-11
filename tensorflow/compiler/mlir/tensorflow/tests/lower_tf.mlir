// RUN: tf-opt %s -test-tf-lower-tf | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: simple_pack
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x5xf32>, %[[ARG1:.*]]: tensor<3x5xf32>
func @simple_pack(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<2x3x5xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) {N = 2 : i64} : (tensor<1x3x5xf32>, tensor<1x3x5xf32>, tensor<i64>) -> tensor<2x3x5xf32>

  %0 = "tf.Pack"(%arg0, %arg1) {N = 2 : i64} : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: pack_with_unranked
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x5xf32>, %[[ARG1:.*]]: tensor<*xf32>
func @pack_with_unranked(%arg0: tensor<?x5xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<-2> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<?x5xf32>, tensor<i64>) -> tensor<?x1x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) {N = 2 : i64} : (tensor<?x1x5xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>

  %0 = "tf.Pack"(%arg0, %arg1) {axis = -2 : i64, N = 2 : i64} : (tensor<?x5xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
