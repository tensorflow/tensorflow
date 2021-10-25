// RUN: xla-opt "-xla-legalize-tf=allow-partial-conversion device-type=XLA_CPU_JIT legalize-chlo=false use-tf2xla-fallback=true prefer-tf2xla=true" %s | FileCheck %s
// RUN: xla-opt "-xla-legalize-tf=allow-partial-conversion device-type=XLA_CPU_JIT legalize-chlo=false prefer-tf2xla=true" %s | FileCheck --check-prefix NOFALLBACK %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: @abs
func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NOT: tf.Abs
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @testBroadcastGradientArgs
func @testBroadcastGradientArgs(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>) {
  // CHECK:     tf.BroadcastGradientArgs
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

// CHECK-LABEL: @acos
func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NOT:  tf.Acos
  %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// NOFALLBACK-LABEL: @xla_svd
func @xla_svd(%arg0: tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
  // NOFALLBACK: XlaSvd
  %s, %u, %v = "tf.XlaSvd"(%arg0) {max_iter = 1, epsilon = 1.0E-09 : f32, precision_config = ""} : (tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
  return %s, %u, %v : tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>
}

}
