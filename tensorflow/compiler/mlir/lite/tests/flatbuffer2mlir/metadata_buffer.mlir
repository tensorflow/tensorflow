// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK: tfl.metadata_buffer = [3 : i32, 7 : i32]
module attributes {tfl.metadata_buffer = [3 : i32, 7 : i32]} {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<3x2xi32>) -> tensor<3x2xi32> {
    %0 = "tfl.add" (%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32>
    func.return %0 : tensor<3x2xi32>
  }
}