// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  // CHECK: func @main(%arg0: tensor<4xf32>)
  // CHECK-NEXT:   return
  // CHECK-NEXT: }

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<4xf32>) -> tensor<4xf32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %2 = "tfl.squared_difference"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  %3 = "tfl.mul"(%0, %2) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  %4 = "tfl.div"(%3, %2) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  %5 = "tfl.exp"(%4) : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  %6 = "tfl.neg"(%5) : (tensor<4xf32>) -> tensor<4xf32> loc("neg")
  return %6 : tensor<4xf32>
}
