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
// RUN: emitters_opt %s -split-input-file -xla-cpu-expand-float-ops | FileCheck %s

func.func @extend(%input: bf16) -> f32 {
  // CHECK-NOT: arith.extf
  %truncated = arith.extf %input : bf16 to f32
  func.return %truncated : f32
}

// -----

func.func @extend_vector(%input: vector<8xbf16>) -> vector<8xf32> {
  // CHECK-NOT: arith.extf
  %truncated = arith.extf %input : vector<8xbf16> to vector<8xf32>
  func.return %truncated : vector<8xf32>
}

// -----

func.func @expm1(%arg0: f64) -> f64 {
  %ret = math.expm1 %arg0 : f64
  return %ret : f64
}

// CHECK-LABEL: @expm1
// CHECK-NOT: math.expm1

// -----

func.func @expm1_vector(%arg0: vector<4xf64>) -> vector<4xf64> {
  %ret = math.expm1 %arg0 : vector<4xf64>
  return %ret : vector<4xf64>
}

// CHECK-LABEL: @expm1_vector
// CHECK-NOT: math.expm1
