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
// RUN: tac-translate -input-mlir -output-mlir -device-specs=NNAPI %s -o - 2>&1 | FileCheck %s

module {
  // CHECK-LABEL: main
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    func.return %0 : tensor<4xf32>
    // CHECK:  [[VAL_0:%.*]] = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
    // CHECK:  [[VAL_1:%.*]] = tfl.mul [[VAL_0]], [[VAL_0]] {fused_activation_function = "NONE"} : tensor<4xf32
  }

  // CHECK-LABEL: pack
  func.func @pack(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1xf32> {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
    func.return %0 : tensor<2x1xf32>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<[2, 1]> : tensor<2xi32>
    // CHECK: %[[CONCAT:.*]] = "tfl.concatenation"(%arg0, %arg1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    // CHECK: %[[VAL_1:.*]] = "tfl.reshape"(%[[CONCAT]], %[[VAL_0]]) : (tensor<2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
    // CHECK: return %[[VAL_1]]
  }
}
