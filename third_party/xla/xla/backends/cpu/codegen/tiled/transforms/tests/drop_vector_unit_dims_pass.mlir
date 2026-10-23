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

// RUN: fusion_compiler_opt %s -xtile-cpu-drop-vector-unit-dims \
// RUN:   -split-input-file | FileCheck %s

func.func @drop_trailing_unit_dim(%arg0: vector<16x1xf32>) -> vector<16x1xf32> {
  %0 = math.isfinite %arg0 : vector<16x1xf32>
  %1 = arith.extui %0 : vector<16x1xi1> to vector<16x1xi8>
  %2 = arith.uitofp %1 : vector<16x1xi8> to vector<16x1xf32>
  return %2 : vector<16x1xf32>
}
// CHECK-LABEL: func.func @drop_trailing_unit_dim(
// CHECK-SAME:     %[[ARG0:.*]]: vector<16x1xf32>) -> vector<16x1xf32> {
// CHECK: %[[CAST_IN:.*]] = vector.shape_cast %[[ARG0]] : vector<16x1xf32> to vector<16xf32>
// CHECK: %[[FINITE:.*]] = math.isfinite %[[CAST_IN]] : vector<16xf32>
// CHECK: %[[EXTUI:.*]] = arith.extui %[[FINITE]] : vector<16xi1> to vector<16xi8>
// CHECK: %[[UITOFP:.*]] = arith.uitofp %[[EXTUI]] : vector<16xi8> to vector<16xf32>
// CHECK: vector.shape_cast %[[UITOFP]] : vector<16xf32> to vector<16x1xf32>

// -----

func.func @drop_leading_unit_dim(%arg0: vector<1x16xf32>, %arg1: vector<1x16xf32>) -> vector<1x16xf32> {
  %0 = arith.addf %arg0, %arg1 : vector<1x16xf32>
  return %0 : vector<1x16xf32>
}
// CHECK-LABEL: func.func @drop_leading_unit_dim(
// CHECK-SAME:     %[[ARG0:.*]]: vector<1x16xf32>, %[[ARG1:.*]]: vector<1x16xf32>) -> vector<1x16xf32> {
// CHECK: %[[CAST_A:.*]] = vector.shape_cast %[[ARG0]] : vector<1x16xf32> to vector<16xf32>
// CHECK: %[[CAST_B:.*]] = vector.shape_cast %[[ARG1]] : vector<1x16xf32> to vector<16xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[CAST_A]], %[[CAST_B]] : vector<16xf32>
// CHECK: vector.shape_cast %[[ADD]] : vector<16xf32> to vector<1x16xf32>

// -----

func.func @drop_leading_and_trailing_unit_dims(%arg0: vector<1x16x1xf32>) -> vector<1x16x1xf32> {
  %0 = math.absf %arg0 : vector<1x16x1xf32>
  return %0 : vector<1x16x1xf32>
}
// CHECK-LABEL: func.func @drop_leading_and_trailing_unit_dims(
// CHECK-SAME:     %[[ARG0:.*]]: vector<1x16x1xf32>) -> vector<1x16x1xf32> {
// CHECK: %[[CAST_IN:.*]] = vector.shape_cast %[[ARG0]] : vector<1x16x1xf32> to vector<16xf32>
// CHECK: %[[ABS:.*]] = math.absf %[[CAST_IN]] : vector<16xf32>
// CHECK: vector.shape_cast %[[ABS]] : vector<16xf32> to vector<1x16x1xf32>

// -----

func.func @no_unit_dims(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.addf %arg0, %arg1 : vector<16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: func.func @no_unit_dims(
// CHECK-SAME:     %[[ARG0:.*]]: vector<16xf32>, %[[ARG1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NOT: vector.shape_cast
// CHECK:     arith.addf %[[ARG0]], %[[ARG1]] : vector<16xf32>

// -----

func.func @transfer_read_non_identity_layout_unit_dim(
    %arg0: memref<2x1x2x1xf32, #xtile.layout<[3, 1, 0, 2]>>, %c0: index) -> vector<2x1x2x1xf32> {
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %pad {in_bounds = [true, true, true, true]} : memref<2x1x2x1xf32, #xtile.layout<[3, 1, 0, 2]>>, vector<2x1x2x1xf32>
  return %0 : vector<2x1x2x1xf32>
}
// CHECK-LABEL: func.func @transfer_read_non_identity_layout_unit_dim(
// CHECK-SAME:     %[[ARG0:.*]]: memref<2x1x2x1xf32, #xtile.layout<[3, 1, 0, 2]>>, %[[C0:.*]]: index) -> vector<2x1x2x1xf32> {
// CHECK-NOT:  memref.subview
// CHECK:      vector.transfer_read %[[ARG0]]

