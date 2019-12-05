// RUN: tf-opt %s -tf-shape-inference -verify-diagnostics | FileCheck %s -dump-input=fail -color

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 130 : i32}} {

// CHECK-LABEL: func @simple_chain
  func @simple_chain(%arg0: tensor<1xf32>) -> tensor<*xf32> {
// CHECK: %[[MUL:.*]] = "tf.Mul"{{.*}} (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[ADD:.*]] = "tf.Add"(%[[MUL]], %[[MUL]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CAST:.*]] = "tf.Cast"(%[[ADD]]) {{.*}} : (tensor<1xf32>) -> tensor<*xf32>
// CHECK: return %[[CAST]] : tensor<*xf32>
    %0 = "tf.Mul"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
    %1 = "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

// CHECK-LABEL: func @simple_chain_with_broadcast
  func @simple_chain_with_broadcast(%arg0: tensor<1xf32>, %arg1: tensor<10xf32>) -> tensor<*xf32> {
// CHECK: %[[MUL:.*]] = "tf.Mul"{{.*}} (tensor<1xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[ADD:.*]] = "tf.Add"(%[[MUL]], %[[MUL]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST:.*]] = "tf.Cast"(%[[ADD]]) {{.*}} : (tensor<10xf32>) -> tensor<*xf32>
// CHECK: return %[[CAST]] : tensor<*xf32>
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<1xf32>, tensor<10xf32>) -> tensor<*xf32>
    %1 = "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

// CHECK-LABEL: func @unknown_op
  func @unknown_op(%arg0: tensor<1xf32>) -> tensor<*xf32> {
// CHECK: %[[MUL:.*]] = "tf.Mul"{{.*}} (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[UNKNOWN:.*]] = "tf.Unknown"(%[[MUL]], %[[MUL]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
// CHECK: return %[[UNKNOWN]] : tensor<*xf32>
    %0 = "tf.Mul"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
    %1 = "tf.Unknown"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

// Tests the case where an inference opportunity relies on folding.

// CHECK-LABEL: func @simple_folding
  func @simple_folding(%arg0: tensor<1x1x1x1xi32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32> {
// CHECK: %[[CST:.*]] = "tf.Const"{{.*}} {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: %[[CONV:.*]] = "tf.Conv2DBackpropInput"(%[[CST]]
// CHECK-SAME: (tensor<4xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK: %[[CAST:.*]] = "tf.Cast"(%[[CONV]]) {{.*}} : (tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32>
// CHECK: return %[[CAST]] : tensor<?x?x?x?xf32>
    %0 = "tf.Shape"(%arg0) : (tensor<1x1x1x1xi32>) -> tensor<4xi32>
    %1 = "tf.Conv2DBackpropInput"(%0, %arg1, %arg1) {
      padding = "VALID", strides = [1, 1, 1, 1]
    } : (tensor<4xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
  }

// Tests the case where an op's shape function returns non-fully-defined shapes.

// CHECK-LABEL: func @op_non_fully_defined_shape_fn
  func @op_non_fully_defined_shape_fn(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<?xi32> {
    // CHECK: tf.BroadcastGradientArgs
    // CHECK-SAME: (tensor<0xi32>, tensor<0xi32>) -> (tensor<?xi32>, tensor<?xi32>)
    %2:2 = "tf.BroadcastGradientArgs"(%arg0, %arg1) {T = "tfdtype$DT_INT32", name = "BroadcastGradientArgs"} : (tensor<0xi32>, tensor<0xi32>) -> (tensor<?xi32>, tensor<?xi32>)
    return %2#0 : tensor<?xi32>
  }

// CHECK-LABEL: func @shape_from_const_input
  func @shape_from_const_input(%arg0: tensor<3x3x32x64xf32>, %arg1: tensor<200x24x24x64xf32>) -> tensor<?x?x?x?xf32> {
    %0 = "tf.Const"() {value = dense<[200, 26, 26, 32]> : tensor<4xi32>} : () -> tensor<4xi32>
    // CHECK: tf.Conv2DBackpropInput
    // CHECK-SAME: (tensor<4xi32>, tensor<3x3x32x64xf32>, tensor<200x24x24x64xf32>) -> tensor<200x26x26x32xf32>
    %1 = "tf.Conv2DBackpropInput"(%0, %arg0, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<4xi32>, tensor<3x3x32x64xf32>, tensor<200x24x24x64xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
  }
}
