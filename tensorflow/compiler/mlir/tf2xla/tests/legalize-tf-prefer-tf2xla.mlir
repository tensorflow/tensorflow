// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion device-type=XLA_CPU_JIT legalize-chlo=false use-tf2xla-fallback=true prefer-tf2xla=true" %s | FileCheck %s
// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion device-type=XLA_CPU_JIT legalize-chlo=false prefer-tf2xla=true" %s | FileCheck --check-prefix NOFALLBACK %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: func @testBroadcastGradientArgs
func.func @testBroadcastGradientArgs(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>) {
  // CHECK:     tf.BroadcastGradientArgs
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

// CHECK-LABEL: @acos
func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NOT:  tf.Acos
  %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// NOFALLBACK-LABEL: @xla_svd
func.func @xla_svd(%arg0: tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
  // NOFALLBACK: XlaSvd
  %s, %u, %v = "tf.XlaSvd"(%arg0) {max_iter = 1, epsilon = 1.0E-09 : f32, precision_config = ""} : (tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
  func.return %s, %u, %v : tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>
}

//===----------------------------------------------------------------------===//
// StridedSlice op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: simple_strided_slice
func.func @simple_strided_slice(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: mhlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  func.return %output : tensor<3x2xf32>
}

}
