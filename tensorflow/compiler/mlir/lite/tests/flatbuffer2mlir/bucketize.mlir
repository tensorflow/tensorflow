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
// Ensure bucketize roundtrip exactly

func.func @main(%arg0: tensor<3x2xf32>) -> tensor<3x2xi32> {
  // CHECK-LABEL: @main
  // CHECK: "tfl.bucketize"(%arg0) <{boundaries = [0.000000e+00 : f32, 1.000000e+01 : f32, 1.000000e+02 : f32]}> : (tensor<3x2xf32>) -> tensor<3x2xi32>
  %0 = "tfl.bucketize"(%arg0) {boundaries = [0.0 : f32, 10.0 : f32, 100.0 : f32]} : (tensor<3x2xf32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}
