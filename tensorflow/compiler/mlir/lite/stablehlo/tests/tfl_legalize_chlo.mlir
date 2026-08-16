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
// RUN: odml-to-stablehlo-opt %s -tfl-legalize-chlo -split-input-file | FileCheck %s --dump-input=fail

// Just assert that pass is properly registered.
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0: tensor<f32>
}
// CHECK-LABEL: main

// -----

func.func @geluWithCustomCallErf(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = stablehlo.constant dense<0.707106769> : tensor<2xf32>
  %2 = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
  %3 = stablehlo.multiply %arg0, %2 : tensor<2xf32>
  %4 = stablehlo.multiply %arg0, %1 : tensor<2xf32>
  %5 = stablehlo.custom_call @mhlo.erf(%4) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<2xf32>) -> tensor<2xf32>
  %6 = stablehlo.add %5, %0 : tensor<2xf32>
  %7 = stablehlo.multiply %3, %6 : tensor<2xf32>
  return %7 : tensor<2xf32>
}

// CHECK-LABEL: geluWithCustomCallErf
// CHECK: "tfl.gelu"
// CHECK-NOT: stablehlo
// CHECK-NOT: chlo

// -----

func.func @geluWithCHLOErf(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = stablehlo.constant dense<0.707106769> : tensor<2xf32>
  %2 = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
  %3 = stablehlo.multiply %arg0, %2 : tensor<2xf32>
  %4 = stablehlo.multiply %arg0, %1 : tensor<2xf32>
  %5 = chlo.erf %4 : tensor<2xf32> -> tensor<2xf32>
  %6 = stablehlo.add %5, %0 : tensor<2xf32>
  %7 = stablehlo.multiply %3, %6 : tensor<2xf32>
  return %7 : tensor<2xf32>
}

// CHECK-LABEL: geluWithCHLOErf
// CHECK: "tfl.gelu"
// CHECK-NOT: stablehlo
// CHECK-NOT: chlo
