// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Ensure bucketize roundtrip exactly

func.func @main(%arg0: tensor<3x2xf32>) -> tensor<3x2xi32> {
  // CHECK-LABEL: @main
  // CHECK: "tfl.bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 1.000000e+01 : f32, 1.000000e+02 : f32]} : (tensor<3x2xf32>) -> tensor<3x2xi32>
  %0 = "tfl.bucketize"(%arg0) {boundaries = [0.0 : f32, 10.0 : f32, 100.0 : f32]} : (tensor<3x2xf32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}
