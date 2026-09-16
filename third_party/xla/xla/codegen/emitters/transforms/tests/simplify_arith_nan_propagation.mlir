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
// RUN: emitters_opt %s -split-input-file -xla-simplify-arith="explicit_nan_propagation=true" -cse -canonicalize | FileCheck %s

module {
  func.func @minimumf_scalar_nan(%arg0: f32, %arg1: f32) -> f32 {
    %min = arith.minimumf %arg0, %arg1 : f32
    return %min : f32
  }
}

// CHECK-LABEL: @minimumf_scalar_nan
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-DAG: %[[ORD:.*]] = arith.cmpf ord, %[[ARG0]], %[[ARG1]] : f32
// CHECK-DAG: %[[CMP:.*]] = arith.cmpf ole, %[[ARG0]], %[[ARG1]] : f32
// CHECK-DAG: %[[SEL:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
// CHECK-DAG: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-NEXT: %[[RET:.*]] = arith.select %[[ORD]], %[[SEL]], %[[NAN]] : f32
// CHECK-NEXT: return %[[RET]] : f32

// -----

module {
  func.func @maximumf_scalar_nan(%arg0: f32, %arg1: f32) -> f32 {
    %max = arith.maximumf %arg0, %arg1 : f32
    return %max : f32
  }
}

// CHECK-LABEL: @maximumf_scalar_nan
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-DAG: %[[ORD:.*]] = arith.cmpf ord, %[[ARG0]], %[[ARG1]] : f32
// CHECK-DAG: %[[CMP:.*]] = arith.cmpf oge, %[[ARG0]], %[[ARG1]] : f32
// CHECK-DAG: %[[SEL:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
// CHECK-DAG: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-NEXT: %[[RET:.*]] = arith.select %[[ORD]], %[[SEL]], %[[NAN]] : f32
// CHECK-NEXT: return %[[RET]] : f32

// -----

module {
  func.func @minimumf_tensor_nan(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %min = arith.minimumf %arg0, %arg1 : tensor<4xf32>
    return %min : tensor<4xf32>
  }
}

// CHECK-LABEL: @minimumf_tensor_nan
// CHECK: arith.constant dense<0x7FC00000> : tensor<4xf32>
