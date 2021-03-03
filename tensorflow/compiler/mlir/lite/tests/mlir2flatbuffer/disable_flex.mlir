// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s

// CHECK: error: 'tf.Div' op is neither a custom op nor a flex op
// CHECK: error: failed while converting: 'main'
// CHECK: Some ops are not supported by the native TFLite runtime
// CHECK: tf.Div {name = "div"}

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_const" () {name = "Const", value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE", name = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // tf.div is the result of conversion to a Flex TF op
  %2 = "tf.Div"(%1, %0) {name = "div"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "tfl.exp"(%2) {name = "exp"} : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
