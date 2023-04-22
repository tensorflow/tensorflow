// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate -input-arrays=squared_difference --experimental-prune-unreachable-nodes-unconditionally --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Tests -input-arrays flag.

func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.squared_difference"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  %2 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  return %2 : tensor<4xf32>

// CHECK-LABEL: main
// CHECK-NOT: tfl.squared_difference
// CHECK: tfl.mul %[[CONST:.*]], %arg0
}
