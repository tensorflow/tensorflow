// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate -output-arrays=mul,exp,div  --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Confirm output-arrays works.

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_input" (%arg0) : (tensor<4xf32>) -> tensor<4xf32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %2 = "tfl.squared_difference"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  // CHECK: %[[MUL:.*]] = tfl.mul
  %3 = "tfl.mul"(%0, %2) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  // CHECK: %[[DIV:.*]] = tfl.div
  %4 = "tfl.div"(%3, %2) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  // CHECK: %[[EXP:.*]] = "tfl.exp"
  %5 = "tfl.exp"(%4) : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  %6 = "tfl.neg"(%5) : (tensor<4xf32>) -> tensor<4xf32> loc("neg")
  // CHECK: return %[[MUL]], %[[EXP]], %[[DIV]]
  return %6 : tensor<4xf32>
}
