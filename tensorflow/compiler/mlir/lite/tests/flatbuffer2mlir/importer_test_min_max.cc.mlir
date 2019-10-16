// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// This only test the exporter and importer are working without min/max quantization parameters.

func @main(tensor<40x37xf32>, tensor<40x37xf32>) -> tensor<40x40xf32> {
^bb0(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>):
  %cst = constant dense<1.0> : tensor<40xf32>
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32> loc("Input")
  %1 = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32> loc("Input")
  %2:2 = "tfl.fully_connected"(%0, %1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  return %2 : tensor<40x40xf32>

// CHECK-LABEL: func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
// CHECK-NEXT: %0 = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32>
// CHECK-NEXT: %1 = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32>
// CHECK-NEXT: %2 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<40xf32>} : () -> tensor<40xf32>
// CHECK-NEXT: %3:2 = "tfl.fully_connected"(%0, %1, %2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>, tensor<40x40xf32>)
// CHECK-NEXT: return %3#0 : tensor<40x40xf32>
// CHECK-NEXT: }
}
