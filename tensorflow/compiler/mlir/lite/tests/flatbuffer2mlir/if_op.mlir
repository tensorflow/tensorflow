// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Confirm function references in if ops are preserved
func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK:   %{{.*}} = "tf.If"(%{{.*}}, %{{.*}}, %{{.*}}) {else_branch = @cond_false, is_stateless = false, then_branch = @cond_true} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.pseudo_input"(%arg1) : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.less"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %3 = "tf.If"(%2, %0, %1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}

func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0 : tensor<*xf32>
}
