// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK-LABEL: main
func.func @main() -> tensor<3x2xi32> {
  // CHECK: "tfl.pseudo_const"() <{value = dense<0> : tensor<3x2xi32>}> {tfl.is_variable} : () -> tensor<3x2xi32>
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3x2xi32>, tfl.is_variable} : () -> tensor<3x2xi32> loc("variable")
  func.return %0 : tensor<3x2xi32>
}