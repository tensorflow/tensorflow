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

// RUN: litert-opt %s -tfl-cast-bf16-ops-to-f32 -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @cast_bf16_single_op
func.func @cast_bf16_single_op(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  // CHECK-DAG: %[[CAST0:.*]] = "tfl.cast"(%arg0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[CAST1:.*]] = "tfl.cast"(%arg1) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK: %[[ADD:.*]] = tfl.add %[[CAST0]], %[[CAST1]] {fused_activation_function = "NONE"} : tensor<2x2xf32>
  // CHECK: %[[CAST_OUT:.*]] = "tfl.cast"(%[[ADD]]) : (tensor<2x2xf32>) -> tensor<2x2xbf16>
  // CHECK: return %[[CAST_OUT]] : tensor<2x2xbf16>
  %0 = "tfl.add"(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %0 : tensor<2x2xbf16>
}

// -----

// CHECK-LABEL: func.func @cast_bf16_chained_ops_removes_redundant_casts
func.func @cast_bf16_chained_ops_removes_redundant_casts(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  // CHECK-DAG: %[[CAST0:.*]] = "tfl.cast"(%arg0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[CAST1:.*]] = "tfl.cast"(%arg1) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK: %[[ADD:.*]] = tfl.add %[[CAST0]], %[[CAST1]] {fused_activation_function = "NONE"} : tensor<2x2xf32>
  // CHECK: %[[CAST2:.*]] = "tfl.cast"(%arg0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK: %[[MUL:.*]] = tfl.mul %[[ADD]], %[[CAST2]] {fused_activation_function = "NONE"} : tensor<2x2xf32>
  // CHECK: %[[CAST_OUT:.*]] = "tfl.cast"(%[[MUL]]) : (tensor<2x2xf32>) -> tensor<2x2xbf16>
  // CHECK: return %[[CAST_OUT]] : tensor<2x2xbf16>
  %0 = "tfl.add"(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  %1 = "tfl.mul"(%0, %arg0) <{fused_activation_function = "NONE"}> : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %1 : tensor<2x2xbf16>
}

// -----

// CHECK-LABEL: func.func @skip_f32_ops
func.func @skip_f32_ops(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK-NOT: tfl.cast
  // CHECK: %[[ADD:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  // CHECK: return %[[ADD]] : tensor<2x2xf32>
  %0 = "tfl.add"(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @skip_call_and_region_ops
func.func @skip_call_and_region_ops(%arg0: tensor<i1>, %arg1: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  // CHECK: %[[CALL:.*]] = {{func\.call|call}} @callee(%arg1) : (tensor<2x2xbf16>) -> tensor<2x2xbf16>
  %0 = func.call @callee(%arg1) : (tensor<2x2xbf16>) -> tensor<2x2xbf16>
  // CHECK: %[[IF:.*]] = "tfl.if"(%arg0) ({
  // CHECK:   %[[CAST1:.*]] = "tfl.cast"(%[[CALL]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK:   %[[CAST2:.*]] = "tfl.cast"(%[[CALL]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // CHECK:   %[[ADD:.*]] = tfl.add %[[CAST1]], %[[CAST2]] {fused_activation_function = "NONE"} : tensor<2x2xf32>
  // CHECK:   %[[CAST_OUT:.*]] = "tfl.cast"(%[[ADD]]) : (tensor<2x2xf32>) -> tensor<2x2xbf16>
  // CHECK:   "tfl.yield"(%[[CAST_OUT]]) : (tensor<2x2xbf16>) -> ()
  // CHECK: }, {
  // CHECK:   "tfl.yield"(%[[CALL]]) : (tensor<2x2xbf16>) -> ()
  // CHECK: }) : (tensor<i1>) -> tensor<2x2xbf16>
  %1 = "tfl.if"(%arg0) ({
    %2 = "tfl.add"(%0, %0) <{fused_activation_function = "NONE"}> : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    "tfl.yield"(%2) : (tensor<2x2xbf16>) -> ()
  }, {
    "tfl.yield"(%0) : (tensor<2x2xbf16>) -> ()
  }) : (tensor<i1>) -> tensor<2x2xbf16>
  return %1 : tensor<2x2xbf16>
}

func.func private @callee(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  return %arg0 : tensor<2x2xbf16>
}
