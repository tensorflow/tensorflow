
func.func @while_cond(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<9> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

func.func @while_body(%arg0: tensor<i32>) -> tensor<i32> {
  %1 = "tf.AddV2"(%arg0, %arg0) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

func.func private @xla_func_0(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.While"(%2) { cond = @while_cond, body = @while_body, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  func.return %1 : tensor<1x3xf32>
}

func.func @main(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %2 = "tf.XlaLaunch"(%arg0, %1) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_0, operand_segment_sizes = array<i32: 0, 2, 0>} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}
