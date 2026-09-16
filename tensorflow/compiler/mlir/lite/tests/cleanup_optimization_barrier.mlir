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
// RUN: litert-opt %s --tfl-cleanup-optimization-barrier --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @cleanup_barrier(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK:           %1 = tfl.add(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK:           return %1 : tensor<2x2xf32>

func.func @cleanup_barrier(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant dense<5.000000e+00> : tensor<f32>
    %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    %1 = stablehlo.optimization_barrier %0 : tensor<2x2xf32>
    %2 = tfl.add(%1, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}
