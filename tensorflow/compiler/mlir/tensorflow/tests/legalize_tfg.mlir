// RUN: tf-opt -tfe-legalize-tfg %s | FileCheck %s

// CHECK: module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 919 : i32}}
module  {
  // CHECK: tf_executor.graph
  tfg.graph #tf_type.version<producer = 919, min_consumer = 12> {
    // CHECK: tf_executor.island wraps "tf.VarHandleOp"() {_mlir_name = "x", _output_shapes = [#tf_type.shape<>], allowed_devices = [], container = "a", device = "/device:CPU:0", dtype = i64, shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i64>>>
    %VarHandleOp, %ctl = VarHandleOp device("/CPU:0") name("x") {_output_shapes = [#tf_type.shape<>], allowed_devices = [], container = "a", dtype = i64, shape = #tf_type.shape<>, shared_name = "x"} : () -> (tensor<!tf_type.resource<tensor<i64>>>)
    // CHECK: tf_executor.island wraps "tf.LegacyCall"(%outputs, %outputs) {_disable_call_shape_inference = true, f = @test_func_name0} : (tensor<!tf_type.resource<tensor<i64>>>, tensor<!tf_type.resource<tensor<i64>>>) -> tensor<*x!tf_type.resource>
    %test_func_name0, %ctl_0 = test_func_name0(%VarHandleOp, %VarHandleOp) name("called") {_disable_call_shape_inference = true, _output_shapes = [#tf_type.shape<*>]} : (tensor<!tf_type.resource<tensor<i64>>>, tensor<!tf_type.resource<tensor<i64>>>) -> (tensor<*x!tf_type.resource>)
    // CHECK: tf_executor.island wraps "tf._Retval"(%outputs_0) {T = !tf_type.resource, _mlir_name = "func_call", index = 0 : i64} : (tensor<*x!tf_type.resource>) -> ()
    %ctl_1 = _Retval(%test_func_name0) name("func_call") {T = !tf_type.resource, index = 0 : i64} : tensor<*x!tf_type.resource>
    // CHECK: tf_executor.fetch
  }

  // CHECK: func @test_func_name0(%arg0: tensor<*x!tf_type.resource> {tf._output_shapes = #tf_type.shape<*>, tf._resource_arg_unique_id = 0 : i64}, %arg1: tensor<*x!tf_type.resource> {tf._output_shapes = #tf_type.shape<*>, tf._resource_arg_unique_id = 0 : i64}) -> tensor<*x!tf_type.resource> attributes {resource_arg_unique_ids_keys = dense<[0, 1]> : tensor<2xi32>, resource_arg_unique_ids_values = dense<0> : tensor<2xi32>, tf._disable_call_shape_inference = true}
  tfg.func @test_func_name0(%test_func_name0: tensor<*x!tf_type.resource> {tf._output_shapes = #tf_type.shape<*>, tf._resource_arg_unique_id = 0 : i64, tfg.name = "test_func_name0"},
                            %test_func_name01: tensor<*x!tf_type.resource> {tf._output_shapes = #tf_type.shape<*>, tf._resource_arg_unique_id = 0 : i64, tfg.name = "test_func_name01"})
       -> (tensor<*x!tf_type.resource> {tfg.dtype = !tf_type.resource, tfg.name = "test_func_name02"})
   // CHECK: tf_executor.graph
   // CHECK: tf_executor.fetch %arg0 : tensor<*x!tf_type.resource>
   // CHECK: return %0 : tensor<*x!tf_type.resource>
   attributes {resource_arg_unique_ids_keys = dense<[0, 1]> : tensor<2xi32>, resource_arg_unique_ids_values = dense<0> : tensor<2xi32>, tf._disable_call_shape_inference = true} {
    return(%test_func_name0) : tensor<*x!tf_type.resource>
  }
}
