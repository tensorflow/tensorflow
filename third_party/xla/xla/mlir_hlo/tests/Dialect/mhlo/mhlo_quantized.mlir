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
// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// -----

// CHECK-LABEL: @uniform_quantized_c1_valid
func.func @uniform_quantized_c1_valid(%arg0: tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.1>> {
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.1>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 0.1>>
}

// -----

func.func @uniform_quantized_c1(%arg0: tensor<2xf32>) {
  // expected-error@+1 {{Expressed type of result expected to be 'f32', but got 'f64'}}
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f64, 0.1>>
  func.return
}

// -----

func.func @uniform_quantized_c1(%arg0: tensor<2x!quant.uniform<i8:f32, 0.1>>) {
  // expected-error@+1 {{Expressed type of result expected to be 'f32', but got 'f64'}}
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<2x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x!quant.uniform<i8:f64, 0.1>>
  func.return
}

// -----

func.func @quantized_ceil_valid(%arg0: tensor<2x!quant.uniform<i8:f32, 0.1>>) {
  %0 = mhlo.ceil %arg0 : tensor<2x!quant.uniform<i8:f32, 0.1>>
  func.return
}

// -----

func.func @quantized_floor_valid(%arg0: tensor<2x!quant.uniform<i8:f32, 0.1>>) {
  %0 = mhlo.floor %arg0 : tensor<2x!quant.uniform<i8:f32, 0.1>>
  func.return
}
