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
// RUN: emitters_opt %s -split-input-file -xla-simplify-arith="fast_min_max=true" -cse -canonicalize | FileCheck %s


module {
  func.func @maximumf(%arg0: f32, %arg1: f32) -> f32 {
    %max = arith.maximumf %arg0, %arg1 : f32
    return %max : f32
  }
}

// CHECK-LABEL: @maximumf
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf uge, %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: return %[[SELECT]] : f32

// -----

module {
  func.func @minimumf(%arg0: f32, %arg1: f32) -> f32 {
    %min = arith.minimumf %arg0, %arg1 : f32
    return %min : f32
  }
}

// CHECK-LABEL: @minimumf
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ule, %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: return %[[SELECT]] : f32
