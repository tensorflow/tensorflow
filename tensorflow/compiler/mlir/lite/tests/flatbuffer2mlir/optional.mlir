// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Test to make sure optional parameters survive a roundtrip

func.func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
// CHECK: [[NONE:%.*]] = "tfl.no_value"() {value} : () -> none
// CHECK: "tfl.fully_connected"(%arg0, %arg1, [[NONE]])
// CHECK-SAME: (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %0:2 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  func.return %0 : tensor<40x40xf32>
}
