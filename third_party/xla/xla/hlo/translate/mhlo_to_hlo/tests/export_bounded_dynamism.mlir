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
// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// CHECK-LITERAL: HloModule main
func.func @main(%arg0: tensor<1x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, 1801]>>) -> tensor<1x16x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, 1801]>> {
  // CHECK: ROOT {{.*}} = f32[1,16,1,<=1801] broadcast
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 2, 3]> : tensor<3xi64>}> : (tensor<1x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, 1801]>>) -> tensor<1x16x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, 1801]>>
  return %0 : tensor<1x16x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, 1801]>>
}

// -----

// CHECK-LITERAL: HloModule main
func.func @main(%arg0: tensor<?x1xf32, #mhlo.type_extensions<bounds = [1801, ?]>>) -> tensor<?x1xf32, #mhlo.type_extensions<bounds = [1801, ?]>> {
  // CHECK: ROOT{{.*}} = f32[<=1801,1] convert
  %0 = mhlo.convert %arg0 : tensor<?x1xf32, #mhlo.type_extensions<bounds = [1801, ?]>>
  return %0 : tensor<?x1xf32, #mhlo.type_extensions<bounds = [1801, ?]>>
}

// -----

// CHECK-LITERAL: HloModule main
func.func @main(%arg0: tensor<1x?x512xf32, #mhlo.type_extensions<bounds = [?, 1800, ?]>>, %arg1: tensor<i32>) -> tensor<1x?x512xf32, #mhlo.type_extensions<bounds = [?, 1800, ?]>> {
  // CHECK: ROOT {{.*}} = f32[1,<=1800,512] set-dimension-size
  %0 = "mhlo.set_dimension_size"(%arg0, %arg1) <{dimension = 1 : i64}> : (tensor<1x?x512xf32, #mhlo.type_extensions<bounds = [?, 1800, ?]>>, tensor<i32>) -> tensor<1x?x512xf32, #mhlo.type_extensions<bounds = [?, 1800, ?]>>
  return %0 : tensor<1x?x512xf32, #mhlo.type_extensions<bounds = [?, 1800, ?]>>
}

// -----

// CHECK-LITERAL: HloModule main
func.func @main(%arg0: tensor<?xf32, #mhlo.type_extensions<bounds = [5]>>) -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>> {
  %0 = mhlo.reshape %arg0 : (tensor<?xf32, #mhlo.type_extensions<bounds = [5]>>) -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
  // CHECK: %[[ARG0:.+]] = f32[<=5] parameter(0)
  // CHECK: f32[1,<=5] reshape(%[[ARG0]])
  return %0 : tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
}
