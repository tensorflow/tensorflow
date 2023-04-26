// RUN: tf-opt "-xla-legalize-tf=device-type=XLA_CPU_JIT allow-partial-conversion=true prefer-tf2xla=true use-tf2xla-fallback=true use-tf2xla-hlo-importer=true" %s -verify-diagnostics -mlir-disable-threading  | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: binary_op
  func.func @binary_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: mhlo.atan2
    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: multiple_return_values
  func.func @multiple_return_values(%arg0: tensor<3xi64>) -> tensor<i64> {
     %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    // CHECK: return %1 : tensor<i64>
    func.return %0#0 : tensor<i64>
  }

  // CHECK-LABEL: constant_parameter
  func.func @constant_parameter(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "tf.Const"() {value = dense<1.42> : tensor<2xf32>} : () -> tensor<2xf32>
    // CHECK: mhlo.atan2 %arg0, %4
    %1 = "tf.Atan2"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: uses_translated_return_type
  func.func @uses_translated_return_type(%arg0: tensor<3xf32>) -> tensor<?xf32> {
    // CHECK: tensor.cast %{{[0-9]+}} : tensor<?xf32, #mhlo.type_extensions<bounds = [3]>> to tensor<?xf32>
    %y, %idx = "tf.Unique"(%arg0) {device = ""} : (tensor<3xf32>) -> (tensor<?xf32>, tensor<3xi32>)
    return %y : tensor<?xf32>
  }

  // CHECK-LABEL: @abs
  func.func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK-NOT: tf.Abs
    %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @testBroadcastGradientArgs
  func.func @testBroadcastGradientArgs(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>) {
    // CHECK:     tf.BroadcastGradientArgs
    %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>)
    func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
  }

  // CHECK-LABEL: @acos
  func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK-NOT:  tf.Acos
    %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: strided_slice_uses_mlir
  func.func @strided_slice_uses_mlir(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
    %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

    // CHECK-NOT: tf.StridedSlice
    %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
        : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
    func.return %output : tensor<3x2xf32>
  }

  // CHECK-LABEL: func @random_uniform_uses_mlir
  func.func @random_uniform_uses_mlir(%arg0: tensor<3xi32>) -> tensor<12x?x64xf32> {
    // CHECK-NOT: tf.RandomUniform
    %0 = "tf.RandomUniform"(%arg0) : (tensor<3xi32>) -> tensor<12x?x64xf32>
    func.return %0 : tensor<12x?x64xf32>
  }
}
