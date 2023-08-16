// RUN: tf-tfrt-opt -split-input-file -tf-executor-to-tfrt-pipeline="target-gpu=true use-gpu-compile-and-execute-op=true func-use-fallback-tensor=true" -tfrt-lower-tf-savedmodel=hoist-invariant-ops=true %s | FileCheck %s --dump-input=fail --dump-input-filter=all

func.func private @xla_func_0(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %1 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: gpurt.compile_and_execute
  // CHECK-SAME: func_name = "xla_func_0"
  // CHECK-SAME: resource_indices = [1]
  %2 = "tf.XlaLaunch"(%arg0, %1) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_0, operandSegmentSizes = array<i32: 0, 2, 0>} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

