// RUN: tf-opt "-xla-legalize-tf=device-type=XLA_CPU_JIT allow-partial-conversion=true prefer-tf2xla=true use-tf2xla-fallback=true use-tf2xla-hlo-importer=true" %s -verify-diagnostics -mlir-disable-threading  | FileCheck %s
// Note: We have to disable threading for this test case because TF2XLA Rewriter
// creates unique function names for the replaced call ops. Running this test in
// multiple threads creates non-deterministic function call names.
// NOTE: Order of test execution can change the translated call name, but its
// syntactic sugar and not an indiciation that the test failed.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: binary_op
  func.func @binary_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: call @translated_tf2xla_kernel_tf.Atan2_0(%arg0, %arg1)
    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: multiple_same_op_unique_function_names
  func.func @multiple_same_op_unique_function_names(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: call @translated_tf2xla_kernel_tf.Atan2_1(%arg0, %arg1)
    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    // CHECK: call @translated_tf2xla_kernel_tf.Atan2_2(%arg0, %arg1)
    %1 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

}
