// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: odml-to-stablehlo-opt %s -split-input-file -stablehlo-optimize -simplify-hlo | FileCheck %s

// Optimizations whose output relies on the folders / canonicalizations that
// MHLO applied implicitly; on StableHLO, `simplify-hlo` (which the pipeline
// runs before legalization) provides them.

// Slices covering the whole tensor in order fuse into a full slice + a
// single-operand concatenate, which simplify back to the input.
// CHECK-LABEL: func.func @testSliceConcat
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3x1x512xf32>
// CHECK-NOT:     stablehlo.slice
// CHECK-NOT:     stablehlo.concatenate
// CHECK:         return %[[ARG0]] : tensor<3x1x512xf32>
func.func @testSliceConcat(%arg0: tensor<3x1x512xf32>) -> tensor<3x1x512xf32> {
  %0 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 1, 1, 512>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %1 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 2, 1, 512>, start_indices = array<i64: 1, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %2 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 3, 1, 512>, start_indices = array<i64: 2, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %r = "stablehlo.concatenate"(%0, %1, %2) <{dimension = 0 : i64}> : (tensor<1x1x512xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<3x1x512xf32>
  func.return %r : tensor<3x1x512xf32>
}
