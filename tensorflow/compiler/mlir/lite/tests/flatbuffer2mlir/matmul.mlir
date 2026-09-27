// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

func.func @main(%arg0: tensor<4x10x15xf32>, %arg1: tensor<4x15x17xf32>) -> tensor<4x10x17xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
  func.return %0:  tensor<4x10x17xf32>

// CHECK-LABEL: main
// CHECK: %[[RESULT0:.*]] =  "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
// CHECK: return %[[RESULT0]]
}

// CHECK-LABEL: testMatmulAsymAttributeTrue
func.func @testMatmulAsymAttributeTrue(%arg0: tensor<4x10x15xf32>, %arg1: tensor<4x15x17xf32>) -> tensor<4x10x17xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = true} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
  func.return %0:  tensor<4x10x17xf32>

// CHECK: %[[RESULT0:.*]] =  "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = true}> : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
// CHECK: return %[[RESULT0]]
}

// CHECK-LABEL: testMatmulAsymAttributeFalse
func.func @testMatmulAsymAttributeFalse(%arg0: tensor<4x10x15xf32>, %arg1: tensor<4x15x17xf32>) -> tensor<4x10x17xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
  func.return %0:  tensor<4x10x17xf32>

// CHECK: %[[RESULT0:.*]] =  "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<4x10x15xf32>, tensor<4x15x17xf32>) -> tensor<4x10x17xf32>
// CHECK: return %[[RESULT0]]
}
