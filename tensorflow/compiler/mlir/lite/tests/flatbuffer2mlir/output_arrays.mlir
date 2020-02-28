// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate -output-arrays=mul,exp,div  --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Confirm output-arrays works.

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.squared_difference"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  // CHECK: %[[MUL:.*]] = tfl.mul
  %2 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  // CHECK: %[[DIV:.*]] = tfl.div
  %3 = "tfl.div"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  // CHECK: %[[EXP:.*]] = "tfl.exp"
  %4 = "tfl.exp"(%3) : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  // tfl.neg should not be pruned
  // CHECK: %[[NEG:.*]] = "tfl.neg"
  %5 = "tfl.neg"(%4) : (tensor<4xf32>) -> tensor<4xf32> loc("neg")
  // CHECK: return %[[MUL]], %[[EXP]], %[[DIV]]
  return %5 : tensor<4xf32>
}
