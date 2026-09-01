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
// RUN: litert-opt %s -inline='default-pipeline=''' | FileCheck %s

// Inline a function that contains only tfl ops.
func.func @func_with_tfl_ops(%arg0 : tensor<2xi32>) -> tensor<2xi32> {
  %0 = "tfl.sub"(%arg0, %arg0) {fused_activation_function = "RELU6"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tfl.add"(%0, %arg0) {fused_activation_function = "RELU6"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %1: tensor<2xi32>
}

// CHECK-LABEL: func @inline_with_arg
// CHECK-SAME:    [[VAL_0:%.*]]: tensor<2xi32>
func.func @inline_with_arg(%arg0 : tensor<2xi32>) -> tensor<2xi32> {
// CHECK-NEXT:  [[VAL_1:%.*]] = tfl.sub [[VAL_0]], [[VAL_0]] {fused_activation_function = "RELU6"} : tensor<2xi32>
// CHECK-NEXT:  [[VAL_2:%.*]] = tfl.add [[VAL_1]], [[VAL_0]] {fused_activation_function = "RELU6"} : tensor<2xi32>
// CHECK-NEXT:  return [[VAL_2]] : tensor<2xi32>
  %0 = func.call @func_with_tfl_ops(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}
