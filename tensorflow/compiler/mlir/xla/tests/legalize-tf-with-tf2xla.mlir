// RUN: tf-opt -xla-legalize-tf-with-tf2xla=device-type=XLA_CPU %s | FileCheck %s --dump-input-on-failure

// INVALID_DEVICE: tf-opt -xla-legalize-tf-with-tf2xla=device-type=INVALID_DEVICE %s | FileCheck %s --dump-input-on-failure

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: abs
// expected-error@+1 {{unsupported device}}
func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[RESULT:.*]] = "xla_hlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

  // return %[[RESULT]]
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: unknown_op
func @unknown_op(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: tf.CustomTestOp
  // expected-remark@+1 {{constant 20}}
  %0 = "tf.CustomTestOp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

  return %0 : tensor<2xf32>
}

// CHECK-LABEL: dynamic_operand
func @dynamic_operand(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: tf.Abs
  // expected-remark@+1 {{lowering requires static shaped operands}}
  %0 = "tf.Abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>

  return %0 : tensor<?xf32>
}

// CHECK-LABEL: multiple_dialect_ops
func @multiple_dialect_ops(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: xla_hlo.neg
  %0 = "xla_hlo.neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: xla_hlo.abs
  %1 = "tf.Abs"(%0) : (tensor<2xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// CHECK-LABEL: binary_op
func @binary_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: xla_hlo.atan2 %arg0, %arg1 : tensor<2xf32>
  %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: binary_op_broadcast
func @binary_op_broadcast(%arg0: tensor<4x1xf32>, %arg1: tensor<4x1x4xf32>) -> tensor<4x4x4xf32> {
  // CHECK: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x1xf32>) -> tensor<4x4x1xf32>
  // CHECK: %[[RESHAPE0:.*]] = "xla_hlo.reshape"(%[[BROADCAST0]]) : (tensor<4x4x1xf32>) -> tensor<4x4xf32>
  // CHECK: %[[UPDATED_ARG0:.*]] = "xla_hlo.broadcast_in_dim"(%[[RESHAPE0]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<4x4x4xf32>

  // CHECK: %[[RESHAPE1:.*]] = "xla_hlo.reshape"(%arg1) : (tensor<4x1x4xf32>) -> tensor<4x4xf32>
  // CHECK: %[[UPDATED_ARG1:.*]] = "xla_hlo.broadcast_in_dim"(%[[RESHAPE1]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<4x4x4xf32>

  // CHECK: %[[RESULT:.*]] = xla_hlo.atan2 %[[UPDATED_ARG0]], %[[UPDATED_ARG1]] : tensor<4x4x4xf32>
  // CHECK: return %[[RESULT]] : tensor<4x4x4xf32>

  %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<4x1xf32>, tensor<4x1x4xf32>) -> tensor<4x4x4xf32>
  return %0: tensor<4x4x4xf32>
}

// CHECK-LABEL: func @ternary_op
func @ternary_op(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: "xla_hlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @convert
func @convert(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: "xla_hlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @constant
func @constant(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[SCALAR_ONE:.*]] = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[ONE:.*]] = "xla_hlo.broadcast_in_dim"(%[[SCALAR_ONE]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<2xf32>
  // CHECK: %[[RESULT:.*]] = xla_hlo.divide %[[ONE]], %arg0 : tensor<2xf32>
  // CHECK: return %[[RESULT]]

  %0 = "tf.Inv"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// TODO(hinsu): Add a test with variant type once one of the ops supporting
// the type is whitelisted. It should be rejected with unsupported type remark.

// TODO(hinsu): Add a test with uint8 type once one of the ops supporting the
// type is whitelisted. Unsigned types are not yet added to the HLO dialect so
// it should return an error. See b/130356985

// TODO(hinsu): Add a test with a valid TF op for which tf2xla kernel is
// available but doesn't support this instance.
}
