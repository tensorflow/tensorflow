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

// Confirm a wide array of attribute survives the round-trip
func.func @main(tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16xf32>):
  // CHECK: "tfl.average_pool_2d"(%{{.*}}) <{filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> loc("avgpool")
  func.return %0 : tensor<1x1x1x16xf32>
}
