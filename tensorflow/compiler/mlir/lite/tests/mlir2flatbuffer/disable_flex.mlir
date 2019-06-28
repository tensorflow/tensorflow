// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s; test ${PIPESTATUS[1]} -eq 1
# CHECK:  loc("disable_flex.mlir":96:8): error: 'tf.div' op is a Flex op but Flex ops are not enabled for emission
# CHECK-NEXT:  Verification failed.

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
