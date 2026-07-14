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
// RUN: litert-opt %s -unfold-large-splat-constant | FileCheck %s

// CHECK-LABEL: @unfold_large_constant_splat
func.func @unfold_large_constant_splat() -> (tensor<10x10xf32>, tensor<1000x1000xf32>) {
  %0 = arith.constant dense<0.00000e+00> : tensor<10x10xf32>
  %1 = arith.constant dense<1.00000e+00> : tensor<1000x1000xf32>
  func.return %0, %1 : tensor<10x10xf32>, tensor<1000x1000xf32>

  // CHECK-DAG: %cst = arith.constant dense<0.000000e+00> : tensor<10x10xf32>
  // CHECK-DAG: %cst_0 = arith.constant dense<1000> : tensor<2xi64>
  // CHECK-DAG: %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %0 = "tfl.fill"(%cst_0, %cst_1) : (tensor<2xi64>, tensor<f32>) -> tensor<1000x1000xf32>
  // CHECK: return %cst, %0 : tensor<10x10xf32>, tensor<1000x1000xf32>
}
