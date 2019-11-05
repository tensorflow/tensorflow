// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s; test ${PIPESTATUS[0]} -ne 0
// CHECK:  error: 'tf.Div' op is neither a custom op nor a flex op
// CHECK:  error: failed while converting: 'main'
// CHECK:  Ops that can be supported by the flex runtime (enabled via setting the -emit-select-tf-ops flag): Div.

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_input" (%arg0) {name = "Input"} : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "tfl.pseudo_const" () {name = "Const", value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %2 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE", name = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // tf.div is the result of conversion to a Flex TF op
  %3 = "tf.Div"(%2, %1) {name = "div"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "tfl.exp"(%3) {name = "exp"} : (tensor<4xf32>) -> tensor<4xf32>
  return %4 : tensor<4xf32>
}
