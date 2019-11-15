// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Test to make sure optional parameters survive a roundtrip

func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
// CHECK: [[NONE:%.*]] = constant unit
// CHECK: "tfl.fully_connected"(%arg0, %arg1, [[NONE]])
// CHECK-SAME: (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  %cst = constant unit
  %0:2 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  return %0 : tensor<40x40xf32>
}
