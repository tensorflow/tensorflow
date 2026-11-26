// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// RUN: xla-opt %s -split-input-file -unsupported-elementwise-to-triton \
// RUN: | FileCheck %s

func.func @converts_tensor_negf_to_subf(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<10xf32>
  // CHECK: %[[SUB:.*]] = arith.subf %[[ZERO]], %arg0 : tensor<10xf32>
  %0 = arith.negf %arg0 : tensor<10xf32>
  // CHECK: return %[[SUB]] : tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

//-----

func.func @converts_scalar_negf_to_subf(%arg0: f32) -> f32 {
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SUB:.*]] = arith.subf %[[ZERO]], %arg0 : f32
  %0 = arith.negf %arg0 : f32
  // CHECK: return %[[SUB]] : f32
  func.return %0 : f32
}

//-----

func.func @converts_tensor_isfinite_f32(%arg0: tensor<10xf32>) -> tensor<10xi1> {
  // CHECK: %[[INF:.*]] = arith.constant dense<0x7F800000> : tensor<10xf32>
  // CHECK: %[[ABS:.*]] = math.absf %arg0 : tensor<10xf32>
  // CHECK: %[[RES:.*]] = arith.cmpf one, %[[ABS]], %[[INF]] : tensor<10xf32>
  %0 = math.isfinite %arg0 : tensor<10xf32>
  // CHECK: return %[[RES]] : tensor<10xi1>
  func.return %0 : tensor<10xi1>
}

//-----

func.func @converts_tensor_isfinite_bf16(%arg0: tensor<8xbf16>) -> tensor<8xi1> {
  // CHECK: %[[INF:.*]] = arith.constant dense<0x7F80> : tensor<8xbf16>
  // CHECK: %[[ABS:.*]] = math.absf %arg0 : tensor<8xbf16>
  // CHECK: %[[RES:.*]] = arith.cmpf one, %[[ABS]], %[[INF]] : tensor<8xbf16>
  %0 = math.isfinite %arg0 : tensor<8xbf16>
  // CHECK: return %[[RES]] : tensor<8xi1>
  func.return %0 : tensor<8xi1>
}

//-----

func.func @converts_tensor_isfinite_f16(%arg0: tensor<8xf16>) -> tensor<8xi1> {
  // CHECK: %[[INF:.*]] = arith.constant dense<0x7C00> : tensor<8xf16>
  // CHECK: %[[ABS:.*]] = math.absf %arg0 : tensor<8xf16>
  // CHECK: %[[RES:.*]] = arith.cmpf one, %[[ABS]], %[[INF]] : tensor<8xf16>
  %0 = math.isfinite %arg0 : tensor<8xf16>
  // CHECK: return %[[RES]] : tensor<8xi1>
  func.return %0 : tensor<8xi1>
}

//-----

func.func @converts_tensor_isfinite_f64(%arg0: tensor<4xf64>) -> tensor<4xi1> {
  // CHECK: %[[INF:.*]] = arith.constant dense<0x7FF0000000000000> : tensor<4xf64>
  // CHECK: %[[ABS:.*]] = math.absf %arg0 : tensor<4xf64>
  // CHECK: %[[RES:.*]] = arith.cmpf one, %[[ABS]], %[[INF]] : tensor<4xf64>
  %0 = math.isfinite %arg0 : tensor<4xf64>
  // CHECK: return %[[RES]] : tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

//-----

func.func @converts_scalar_isfinite(%arg0: f32) -> i1 {
  // CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
  // CHECK: %[[ABS:.*]] = math.absf %arg0 : f32
  // CHECK: %[[RES:.*]] = arith.cmpf one, %[[ABS]], %[[INF]] : f32
  %0 = math.isfinite %arg0 : f32
  // CHECK: return %[[RES]] : i1
  func.return %0 : i1
}

//-----

func.func @leaves_fp8_isfinite_unchanged(%arg0: tensor<8xf8E4M3FN>) -> tensor<8xi1> {
  // CHECK: %[[RES:.*]] = math.isfinite %arg0 : tensor<8xf8E4M3FN>
  %0 = math.isfinite %arg0 : tensor<8xf8E4M3FN>
  // CHECK: return %[[RES]] : tensor<8xi1>
  func.return %0 : tensor<8xi1>
}
