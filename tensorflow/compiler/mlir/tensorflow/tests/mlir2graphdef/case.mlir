// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 486 : i32}} {
  func @main() {
    tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Case"(%outputs_0, %outputs) {Tin = [i32], Tout = [i32], _lower_using_switch_merge = true, _read_only_resource_inputs = [], branches = [@indexed_case_branch0_40, @indexed_case_branch1_50], device = "", is_stateless = true, output_shapes = [#tf_type.shape<>]} : (tensor<i32>, tensor<i32>) -> tensor<*xi32> loc("stateless_case")
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Identity"(%outputs_2) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Case"(%outputs_0, %outputs) {Tin = [i32], Tout = [i32], _lower_using_switch_merge = true, _read_only_resource_inputs = [], branches = [@indexed_case_branch0_40, @indexed_case_branch1_50], device = "", is_stateless = false, output_shapes = [#tf_type.shape<>]} : (tensor<i32>, tensor<i32>) -> tensor<*xi32> loc("regular_case")
      tf_executor.fetch
    }
    return
  }

  func private @indexed_case_branch0_40(%arg0: tensor<i32>) -> tensor<*xi32> {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.AddV2"(%arg0, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.fetch %outputs_0 : tensor<*xi32>
    }
    return %0 : tensor<*xi32>
  }

  func private @indexed_case_branch1_50(%arg0: tensor<i32>) -> tensor<*xi32> {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<2> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.AddV2"(%arg0, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.fetch %outputs_0 : tensor<*xi32>
    }
    return %0 : tensor<*xi32>
  }
}

// CHECK: name: "stateless_case"
// CHECK-NEXT: "StatelessCase"
// CHECK: name: "regular_case"
// CHECK-NEXT: "Case"
