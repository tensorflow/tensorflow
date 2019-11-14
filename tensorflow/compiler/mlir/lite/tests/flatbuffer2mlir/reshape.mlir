// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Confirm we can extract type info from reshape

func @main() -> tensor<2x2xf32> {
  // CHECK: %[[cst:.*]] = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %{{.*}} = "tfl.reshape"(%{{.*}}, %[[cst]]) : (tensor<4xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.reshape" (%0, %cst) : (tensor<4xf32>, tensor<2xi32>) -> tensor<2x2xf32> loc("reshape")
  return %1 : tensor<2x2xf32>
}
