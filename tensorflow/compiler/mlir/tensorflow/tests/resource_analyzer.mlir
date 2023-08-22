// RUN: tf-opt -verify-diagnostics -allow-unregistered-dialect \
// RUN:        -split-input-file -tf-resource-analyzer-test %s \
// RUN:   | FileCheck %s

// TODO(b/269548549): Add tests to cover more patterns.

// Test that VarHandleOp is not marked as "potentially written" if it is not
// assigned inside the function called by "tf.BatchFunction".

module {
// CHECK-LABEL: @serving_default
  func.func @serving_default() -> (tensor<*xi32>) {
    // expected-remark@below {{device: "", container: "", shared_name: "var_0", is_potentially_written: false}}
    %0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xi32>>>
    %1 = "tf.BatchFunction"(%0) {
        f = @called_by_batch_func,
        operandSegmentSizes = array<i32: 1, 0>,
        batch_timeout_micros = 1000,
        max_batch_size = 8,
        num_batch_threads = 2
      } : (tensor<!tf_type.resource<tensor<2xi32>>>) -> tensor<*xi32>
    return %1 : tensor<*xi32>
  }

  func.func private @called_by_batch_func(%arg: tensor<*x!tf_type.resource>) -> tensor<?xi32> {
    %0 = "tf.ReadVariableOp"(%arg) : (tensor<*x!tf_type.resource>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Test that VarHandleOp is marked as "potentially written" if it is
// assigned inside the function called by "tf.BatchFunction".

module {
// CHECK-LABEL: @serving_default
  func.func @serving_default() -> (tensor<*xi32>) {
    // expected-remark@below {{device: "", container: "", shared_name: "var_0", is_potentially_written: true}}
    %0 = "tf.VarHandleOp"() {shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xi32>>>
    %1 = "tf.BatchFunction"(%0) {
        f = @called_by_batch_func_assign,
        operandSegmentSizes = array<i32: 1, 0>,
        batch_timeout_micros = 1000,
        max_batch_size = 8,
        num_batch_threads = 2
      } : (tensor<!tf_type.resource<tensor<2xi32>>>) -> tensor<*xi32>
    return %1 : tensor<*xi32>
  }

  func.func private @called_by_batch_func_assign(%arg: tensor<*x!tf_type.resource>) -> tensor<?xi32> {
    %0 = "tf.Const"() {value = dense<4> : tensor<2xi32>} : () -> tensor<2xi32>
    "tf.AssignVariableOp"(%arg, %0) : (tensor<*x!tf_type.resource>, tensor<2xi32>) -> ()
    %1 = "tf.ReadVariableOp"(%arg) : (tensor<*x!tf_type.resource>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
