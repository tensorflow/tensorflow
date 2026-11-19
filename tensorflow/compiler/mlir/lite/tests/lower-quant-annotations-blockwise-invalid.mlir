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
// RUN: litert-opt %s -split-input-file -verify-diagnostics --tfl-lower-quant-annotations

// Blockwise scales and zero points must have one axis per weight dimension,
// each of extent 1 or the number of blocks along it.

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

func.func @uniform_zero_point_with_wrong_extent() -> tensor<2x4xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  // expected-error @+1 {{blockwise zero point dimension 0 is 3, expected 1 or 2 (the number of blocks)}}
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<-0.5> : tensor<3x1xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

func.func @zero_point_with_wrong_rank() -> tensor<2x4xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  // expected-error @+1 {{blockwise zero point has rank 1 but the weight has rank 2}}
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<[0.0, 1.0]> : tensor<2xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

func.func @scale_with_wrong_rank() -> tensor<2x4xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  // expected-error @+1 {{blockwise scale has rank 1 but the weight has rank 2}}
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[0.5, 0.25]> : tensor<2xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
