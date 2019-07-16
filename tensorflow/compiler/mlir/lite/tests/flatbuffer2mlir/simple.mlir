// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s

func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  // CHECK: func @main(%arg0: tensor<3x2xi32>) {
  // CHECK-NEXT:   return
  // CHECK-NEXT: }

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32> loc("Const")
  %2 = "tfl.sub" (%0, %1) {fused_activation_function = "RELU6"} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("sub")
  %3 = "std.constant" () {value = dense<10> : tensor<i32>} : () -> tensor<i32> loc("Const2")
  %4 = "tfl.add" (%3, %2) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("add")
  return %4 : tensor<3x2xi32>
}
