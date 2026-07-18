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
