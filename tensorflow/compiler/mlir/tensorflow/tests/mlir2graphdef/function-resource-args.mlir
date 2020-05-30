// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s --dump-input-on-failure

func @main() -> tensor<*x!tf.resource> attributes {tf.entry_function = {inputs = "", outputs = "func_call"}} {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.VarHandleOp"() {container = "a", device = "/CPU:0", dtype = i64, shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<i64>>> loc("x")
    %outputs_0, %control_1 = tf_executor.island wraps "tf.LegacyCall"(%outputs, %outputs) {_disable_call_shape_inference = true, f = @test_func_name0} : (tensor<!tf.resource<tensor<i64>>>, tensor<!tf.resource<tensor<i64>>>) -> tensor<*x!tf.resource> loc("called")
    tf_executor.fetch %outputs_0 : tensor<*x!tf.resource>
  }
  return %0 : tensor<*x!tf.resource>
}
func @test_func_name0(%arg0: tensor<*x!tf.resource> {tf._resource_arg_unique_id = 0 : i64}, %arg1: tensor<*x!tf.resource> {tf._resource_arg_unique_id = 0 : i64}) -> tensor<*x!tf.resource> attributes {tf._disable_call_shape_inference = true} {
  %0 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*x!tf.resource>
  }
  return %0 : tensor<*x!tf.resource>
}

// Check that the `tf._resource_arg_unique_id` argument attributes of
// test_func_name0 are propagated to the function's arg_attr and
// resource_arg_unique_id.

// CHECK:      name: "x"
// CHECK:      op: "VarHandleOp"

// CHECK:      name: "func_call"
// CHECK:      input: "called"

// CHECK:      library
// CHECK:        function
// CHECK:          signature
// CHECK:            input_arg
// CHECK:              type: DT_RESOURCE
// CHECK:            input_arg
// CHECK:              type: DT_RESOURCE
// CHECK:            output_arg
// CHECK:              type: DT_RESOURCE
// CHECK:          ret

// Check _resource_arg_unique_id for each argument. Since they alias each other,
// both values are 0.
// CHECK:          arg_attr
// CHECK-NEXT:       key: 0
// CHECK-NEXT:       value
// CHECK:             key: "_resource_arg_unique_id"
// CHECK-NEXT:        value
// CHECK-NEXT:          i: 0
// CHECK:          arg_attr
// CHECK-NEXT:       key: 1
// CHECK-NEXT:       value
// CHECK:              key: "_resource_arg_unique_id"
// CHECK-NEXT:         value
// CHECK-NEXT:           i: 0

// Check resource_arg_unique_id for each argument. Since they alias each other,
// both values are 0.
// CHECK:          resource_arg_unique_id
// CHECK-NEXT:       key: 0
// CHECK-NEXT:       value: 0
// CHECK:          resource_arg_unique_id
// CHECK-NEXT:       key: 1
// CHECK-NEXT:       value: 0
