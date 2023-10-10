// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

func.func @main(%arg0: tensor<4x10x15xf32>, %arg1: tensor<4x15x17xf32>) -> tensor<4x10x17xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
  func.return %0:  tensor<4x10x17xf32>

// CHECK-LABEL: main
// CHECK: %[[RESULT0:.*]] =  "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
// CHECK: return %[[RESULT0]]
}

// CHECK-LABEL: testMatmulAsymAttributeTrue
func.func @testMatmulAsymAttributeTrue(%arg0: tensor<4x10x15xf32>, %arg1: tensor<4x15x17xf32>) -> tensor<4x10x17xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = true} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
  func.return %0:  tensor<4x10x17xf32>

// CHECK: %[[RESULT0:.*]] =  "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = true} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
// CHECK: return %[[RESULT0]]
}

// CHECK-LABEL: testMatmulAsymAttributeFalse
func.func @testMatmulAsymAttributeFalse(%arg0: tensor<4x10x15xf32>, %arg1: tensor<4x15x17xf32>) -> tensor<4x10x17xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
  func.return %0:  tensor<4x10x17xf32>

// CHECK: %[[RESULT0:.*]] =  "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
// CHECK: return %[[RESULT0]]
}
