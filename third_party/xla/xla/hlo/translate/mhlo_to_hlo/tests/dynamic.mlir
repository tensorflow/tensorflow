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
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK: HloModule main, entry_computation_layout={(s64[<=4,1]{1,0})->s64[1,<=4]{1,0}}
func.func @main(%arg0: tensor<?x1xi64, #mhlo.type_extensions<bounds = [4, ?]>>) -> tensor<1x?xi64, #mhlo.type_extensions<bounds = [?, 4]>> {
  %0 = mhlo.constant dense<1> : tensor<1xi32>
  %1 = "mhlo.get_dimension_size"(%arg0) <{dimension = 0 : i64}> : (tensor<?x1xi64, #mhlo.type_extensions<bounds = [4, ?]>>) -> tensor<i32>
  %2 = mhlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
  %3 = "mhlo.concatenate"(%0, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %4 = mhlo.dynamic_reshape %arg0, %3 : (tensor<?x1xi64, #mhlo.type_extensions<bounds = [4, ?]>>, tensor<2xi32>) -> tensor<1x?xi64, #mhlo.type_extensions<bounds = [?, 4]>>
  func.return %4 : tensor<1x?xi64, #mhlo.type_extensions<bounds = [?, 4]>>
  //      CHECK: %[[ARG0:.*]] = s64[<=4,1] parameter(0)
  // CHECK-NEXT: %[[SIZE0x1:.*]] = s32[1] constant({1})
  // CHECK-NEXT: %[[SIZE1:.*]] = s32[] get-dimension-size(%[[ARG0]]), dimensions={0}
  // CHECK-NEXT: %[[SIZE1x1:.*]] = s32[1] reshape(%[[SIZE1]])
  // CHECK-NEXT: %[[SHAPE:.*]] = s32[2] concatenate(%[[SIZE0x1]], %[[SIZE1x1]]), dimensions={0}
  // CHECK-NEXT: %[[SHAPE0x1:.*]] = s32[1] slice(%[[SHAPE]]), slice={[0:1]}
  // CHECK-NEXT: %[[SHAPE0:.*]] = s32[] reshape(%[[SHAPE0x1]])
  // CHECK-NEXT: %[[SHAPE1x1:.*]] = s32[1] slice(%[[SHAPE]]), slice={[1:2]}
  // CHECK-NEXT: %[[SHAPE1:.*]] = s32[] reshape(%[[SHAPE1x1]])
  // CHECK-NEXT: ROOT %dynamic-reshape.1 = s64[1,<=4] dynamic-reshape(%[[ARG0]], %[[SHAPE0]], %[[SHAPE1]])
}
