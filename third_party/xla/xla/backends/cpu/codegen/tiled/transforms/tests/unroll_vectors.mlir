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

// RUN: fusion_compiler_opt %s --xtile-cpu-unroll-vectors --canonicalize \
// RUN:   -split-input-file | FileCheck %s

func.func @unroll_2d_elementwise(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %dest: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %pad {in_bounds = [true, true]} : tensor<2x4xf32>, vector<2x4xf32>
  %1 = vector.transfer_read %arg1[%c0, %c0], %pad {in_bounds = [true, true]} : tensor<2x4xf32>, vector<2x4xf32>
  %2 = arith.addf %0, %1 : vector<2x4xf32>
  %3 = vector.transfer_write %2, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, tensor<2x4xf32>
  return %3 : tensor<2x4xf32>
}
// CHECK-LABEL: func.func @unroll_2d_elementwise(
// CHECK-COUNT-2: vector.transfer_read %{{.*}} : tensor<2x4xf32>, vector<1x4xf32>
// CHECK-COUNT-2: arith.addf %{{.*}}, %{{.*}} : vector<1x4xf32>
// CHECK-COUNT-2: vector.transfer_write %{{.*}} : vector<1x4xf32>, tensor<2x4xf32>

// -----

func.func @unroll_2d_math_is_finite(%arg0: tensor<2x4xf32>, %dest: tensor<2x4xi1>) -> tensor<2x4xi1> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %pad {in_bounds = [true, true]} : tensor<2x4xf32>, vector<2x4xf32>
  %1 = math.isfinite %0 : vector<2x4xf32>
  %2 = vector.transfer_write %1, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xi1>, tensor<2x4xi1>
  return %2 : tensor<2x4xi1>
}
// CHECK-LABEL: func.func @unroll_2d_math_is_finite(
// CHECK-COUNT-2: vector.transfer_read %{{.*}} : tensor<2x4xf32>, vector<1x4xf32>
// CHECK-COUNT-2: math.isfinite %{{.*}} : vector<1x4xf32>
// CHECK-COUNT-2: vector.transfer_write %{{.*}} : vector<1x4xi1>, tensor<2x4xi1>

// -----

func.func @unroll_3d_elementwise(%arg0: tensor<2x2x4xf32>, %dest: tensor<2x2x4xf32>) -> tensor<2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} : tensor<2x2x4xf32>, vector<2x2x4xf32>
  %1 = math.absf %0 : vector<2x2x4xf32>
  %2 = vector.transfer_write %1, %dest[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<2x2x4xf32>, tensor<2x2x4xf32>
  return %2 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @unroll_3d_elementwise(
// CHECK-COUNT-4: vector.transfer_read %{{.*}} : tensor<2x2x4xf32>, vector<1x1x4xf32>
// CHECK-COUNT-4: math.absf %{{.*}} : vector<1x1x4xf32>
// CHECK-COUNT-4: vector.transfer_write %{{.*}} : vector<1x1x4xf32>, tensor<2x2x4xf32>

// -----

func.func @unroll_constant_mask_partial(%dest: tensor<2x4xi1>) -> tensor<2x4xi1> {
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 2] : vector<2x4xi1>
  %1 = vector.transfer_write %0, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xi1>, tensor<2x4xi1>
  return %1 : tensor<2x4xi1>
}
// CHECK-LABEL: func.func @unroll_constant_mask_partial(
// CHECK-DAG:     %[[M0:.*]] = vector.constant_mask [1, 2] : vector<1x4xi1>
// CHECK-DAG:     %[[M1:.*]] = arith.constant dense<false> : vector<1x4xi1>
// CHECK:         vector.transfer_write %[[M0]]
// CHECK:         vector.transfer_write %[[M1]]

// -----

func.func @no_unroll_1d(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %dest: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %pad {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
  %1 = vector.transfer_read %arg1[%c0], %pad {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
  %2 = arith.addf %0, %1 : vector<4xf32>
  %3 = vector.transfer_write %2, %dest[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
  return %3 : tensor<4xf32>
}
// CHECK-LABEL: func.func @no_unroll_1d(
// CHECK-COUNT-2: vector.transfer_read %{{.*}} : tensor<4xf32>, vector<4xf32>
// CHECK-COUNT-1: arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-1: vector.transfer_write %{{.*}} : vector<4xf32>, tensor<4xf32>

// -----

func.func @no_unroll_unit_leading(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>, %dest: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %pad {in_bounds = [true, true]} : tensor<1x4xf32>, vector<1x4xf32>
  %1 = vector.transfer_read %arg1[%c0, %c0], %pad {in_bounds = [true, true]} : tensor<1x4xf32>, vector<1x4xf32>
  %2 = arith.addf %0, %1 : vector<1x4xf32>
  %3 = vector.transfer_write %2, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
  return %3 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @no_unroll_unit_leading(
// CHECK-COUNT-2: vector.transfer_read %{{.*}} : tensor<1x4xf32>, vector<1x4xf32>
// CHECK-COUNT-1: arith.addf %{{.*}}, %{{.*}} : vector<1x4xf32>
// CHECK-COUNT-1: vector.transfer_write %{{.*}} : vector<1x4xf32>, tensor<1x4xf32>
