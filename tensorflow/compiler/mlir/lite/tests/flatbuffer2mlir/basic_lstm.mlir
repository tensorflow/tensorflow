// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Ensure basic_lstm roundtrip exactly

func @main(%arg0: tensor<1x384xf32>, %arg1: tensor<1x96xf32>, %arg2: tensor<384x480xf32>, %arg3: tensor<384xf32>, %arg4: tensor<1x96xf32>) -> tensor<1x96xf32> {
// CHECK-LABEL: @main
// CHECK: "tfl.basic_lstm"({{.*}}) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "RELU", kernel_type = "BASIC", proj_clip = 2.000000e+00 : f32} : (tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> (tensor<1x96xf32>, tensor<1x96xf32>, tensor<1x480xf32>, tensor<1x384xf32>)

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<1x384xf32>) -> tensor<1x384xf32>
  %1 = "tfl.pseudo_input" (%arg1) : (tensor<1x96xf32>) -> tensor<1x96xf32>
  %2 = "tfl.pseudo_input" (%arg2) : (tensor<384x480xf32>) -> tensor<384x480xf32>
  %3 = "tfl.pseudo_input" (%arg3) : (tensor<384xf32>) -> tensor<384xf32>
  %4 = "tfl.pseudo_input" (%arg4) : (tensor<1x96xf32>) -> tensor<1x96xf32>
  %5:4 = "tfl.basic_lstm"(%0, %1, %2, %3, %4) {fused_activation_function = "RELU", cell_clip = 1.0 : f32, proj_clip = 2.0 : f32} : (tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> (tensor<1x96xf32>, tensor<1x96xf32>, tensor<1x480xf32>, tensor<1x384xf32>)
  return %5#0 : tensor<1x96xf32>
}
