// RUN: tf-opt "-xla-legalize-tf=device-type=XLA_CPU_JIT allow-partial-conversion=true prefer-tf2xla=true use-tf2xla-fallback=true use-tf2xla-hlo-importer=true" %s -verify-diagnostics -mlir-disable-threading  | FileCheck %s
// Note: We have to disable threading for this test case because TF2XLA Rewriter
// creates unique function names for the replaced call ops. Running this test in
// multiple threads creates non-deterministic function call names.
// NOTE: Order of test execution can change the translated call name, but its
// syntactic sugar and not an indiciation that the test failed.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: binary_op
  func.func @binary_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: call @tf2xla_rewriter.tf.Atan2.5.0(%arg0, %arg1)
    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: multiple_same_op_unique_function_names
  func.func @multiple_same_op_unique_function_names(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: call @tf2xla_rewriter.tf.Atan2.5.1(%arg0, %arg1)
    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    // CHECK: call @tf2xla_rewriter.tf.Atan2.5.2(%arg0, %arg1)
    %1 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: multiple_return_values
  func.func @multiple_return_values(%arg0: tensor<3xi64>) -> tensor<i64> {
    // CHECK: call @tf2xla_rewriter.tf.Unpack.9.0(%arg0) : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
     %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    func.return %0#0 : tensor<i64>
  }

  // CHECK-LABEL: constant_parameter
  func.func @constant_parameter(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "tf.Const"() {value = dense<1.42> : tensor<2xf32>} : () -> tensor<2xf32>
    // CHECK: call @tf2xla_rewriter.tf.Atan2.9.0(%arg0, %0)
    %1 = "tf.Atan2"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: uses_translated_return_type
  func.func @uses_translated_return_type(%arg0: tensor<3xf32>) -> tensor<?xf32> {
    // CHECK: call @tf2xla_rewriter.tf.Unique.126.0(%arg0) : (tensor<3xf32>) -> (tensor<?xf32, #mhlo.type_extensions<bounds = [3]>>, tensor<3xi32>)
    %y, %idx = "tf.Unique"(%arg0) {device = ""} : (tensor<3xf32>) -> (tensor<?xf32>, tensor<3xi32>)
    return %y : tensor<?xf32>
  }

}
