// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Confirm function references in if ops are preserved. 
// TODO(b/137395003): Currently we export both tf.If and tfl.functional_if
// as the same tflite builtin op. Delete tf.If case once legalization
// is supported.
func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
  // CHECK: %1 = "tf.If"(%0, %arg0, %arg1) <{else_branch = @tfl_cond_false, is_stateless = false, then_branch = @tfl_cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  // CHECK: %2 = "tf.If"(%0, %arg0, %arg1) <{else_branch = @tf_cond_false, is_stateless = false, then_branch = @tf_cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %1 = "tfl.functional_if"(%0, %arg0, %arg1) {else_branch = @tfl_cond_false, then_branch = @tfl_cond_true} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tf.If"(%0, %arg0, %arg1) {else_branch = @tf_cond_false, then_branch = @tf_cond_true, is_stateless = false} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %1, %2 : tensor<1xf32>, tensor<1xf32>
}

func.func @tfl_cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @tfl_cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @tf_cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @tf_cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
