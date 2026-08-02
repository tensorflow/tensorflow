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
// RUN: not xla-translate -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: 'stablehlo.composite' op CompositeOp with regions not supported in StableHLO -> HLO conversion
  %0 = "mhlo.composite"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) {
    name = "foo.bar",
    composite_attributes = {},
    decomposition = @add,
    version = 1 : i32
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @add(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}
