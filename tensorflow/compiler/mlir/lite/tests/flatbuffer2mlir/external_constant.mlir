// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --use-external-constant - -o - | FileCheck %s
// Ensure that `tfl.external_const` is imported when the flag `-use-external-constant` is enabled.

func @main(tensor<40x37xf32>, tensor<40x37xf32>) -> tensor<40x40xf32> {
^bb0(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>):
  %cst = arith.constant dense<1.0> : tensor<40xf32>
  %0:2 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  return %0 : tensor<40x40xf32>

// CHECK-LABEL: func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32>
// CHECK:      %[[CONST:[0-9]+]] = "tfl.external_const"() {buffer_index = 3 : i32} : () -> tensor<40xf32>
// CHECK-NEXT: %[[FULL:[0-9]+]]:2 = "tfl.fully_connected"(%arg0, %arg1, %[[CONST]]) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}
// CHECK-NEXT: return %[[FULL]]#0
}
