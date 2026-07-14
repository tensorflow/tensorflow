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
// RUN: hlo-translate -mlir-to-hlo -split-input-file %s | FileCheck %s

// Part of HLO lowering includes MLIR preprocessing for things that are allowed
// in MLIR HLO but not in HLO.

// CHECK-LABEL: main
// [[ARG_0:%.*]] = f32[1,2,3] parameter(0)
// [[TRANSPOSE:%.*]] = f32[2,3,1] transpose([[ARG_0]]), dimensions={1,2,0}
// [[BROADCAST_0:%.*]] = f32[2,3,1,1] broadcast([[TRANSPOSE]]), dimensions={0,1,3}
// [[RESHAPE:%.*]] = f32[2,3,1] reshape([[BROADCAST_0]])
// ROOT %[[BROADCAST_1:%.*]] = f32[2,3,1,10] broadcast([[RESHAPE]]), dimensions={0,1,2}

func.func @main(%arg0: tensor<1x2x3xf32>) -> tensor<2x3x1x10xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [3, 0, 1] : (tensor<1x2x3xf32>) -> tensor<2x3x1x10xf32>
  return %0 : tensor<2x3x1x10xf32>
}
