// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

func.func @main(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<1x32x42x128xf32>, %arg2: tensor<4xi32>) -> tensor<1x64x84x32xf32> {
  %0 = "tfl.custom"(%arg0, %arg1, %arg2) {custom_code = "Convolution2DTransposeBias", custom_option = #tfl<const_bytes : "0x010000000200000002000000">} : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<4xi32>) -> tensor<1x64x84x32xf32>
  func.return %0 : tensor<1x64x84x32xf32>
}
// CHECK-LABEL: main
// CHECK: "tfl.custom"(%arg0, %arg1, %arg2) <{custom_code = "Convolution2DTransposeBias", custom_option = #tfl<const_bytes : "0x010000000200000002000000">}> : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<4xi32>) -> tensor<1x64x84x32xf32>
