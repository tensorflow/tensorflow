// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK: func @main(%arg0: tensor<?x19x19x3xf32>) -> tensor<?x9x9x4xf32>
func @main(%arg0: tensor<?x19x19x3xf32>) -> tensor<?x9x9x4xf32> {
  %cst = constant dense<1.0> : tensor<4xf32>
  %cst_3 = constant dense<2.0> : tensor<4x3x3x3xf32>
  %0 = "tfl.conv_2d"(%arg0, %cst_3, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU6", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x19x19x3xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<?x9x9x4xf32>
  return %0 : tensor<?x9x9x4xf32>
}
