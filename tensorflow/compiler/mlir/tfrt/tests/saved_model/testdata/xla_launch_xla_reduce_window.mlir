
func.func private @sum_reducer(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.AddV2"(%arg0, %arg1) {device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @xla_func_0(%arg0: tensor<7xf32>, %arg1: tensor<f32>) -> tensor<10xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_3 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.XlaReduceWindow"(%arg0, %arg1, %cst_0, %cst_1, %cst_2, %cst_3, %cst) {computation = @sum_reducer} : (tensor<7xf32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

func.func @main(%arg0: tensor<7xf32>) -> tensor<10xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %2 = "tf.XlaLaunch"(%arg0, %1) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_0, operand_segment_sizes = array<i32: 0, 2, 0>} : (tensor<7xf32>, tensor<f32>) -> tensor<10xf32>
  func.return %2 : tensor<10xf32>
}
