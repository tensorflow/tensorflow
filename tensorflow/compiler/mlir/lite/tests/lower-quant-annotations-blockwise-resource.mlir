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
// RUN: litert-opt %s -split-input-file --tfl-lower-quant-annotations | FileCheck %s

// Static blockwise fake quantizers of float16 / bfloat16 weights stored as
// dense resources (as large constants are): the weights are read and
// quantized at compile time. Centered int2: one block per row, zero point
// -0.5; row 0 holds (q + 0.5) * 0.5 for q = [-2, -1, 0, 1], row 1
// (q + 0.5) * 0.25 for q = [1, 0, -1, -2].

func.func private @fq_impl_f16(%arg0: tensor<2x4xf16>) -> tensor<2x4xf16> {
  return %arg0 : tensor<2x4xf16>
}

// CHECK-LABEL: func.func @blockwise_f16_resource_weight
func.func @blockwise_f16_resource_weight() -> tensor<2x4xf16> {
  // CHECK: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -2]]> : tensor<2x4xi2>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %{{.+}}, %{{.+}}) {{.*}} -> tensor<2x4xf16>
  %w = stablehlo.constant dense_resource<w_f16> : tensor<2x4xf16>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<-0.5> : tensor<2x1xf32>}, decomposition = @fq_impl_f16} : (tensor<2x4xf16>) -> tensor<2x4xf16>
  return %0 : tensor<2x4xf16>
}

{-#
  dialect_resources: {
    builtin: {
      w_f16: "0x0200000000BA00B40034003A0036003000B000B6"
    }
  }
#-}

// -----

func.func private @fq_impl_bf16(%arg0: tensor<2x4xbf16>) -> tensor<2x4xbf16> {
  return %arg0 : tensor<2x4xbf16>
}

// CHECK-LABEL: func.func @blockwise_bf16_resource_weight
func.func @blockwise_bf16_resource_weight() -> tensor<2x4xbf16> {
  // CHECK: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -2]]> : tensor<2x4xi2>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %{{.+}}, %{{.+}}) {{.*}} -> tensor<2x4xbf16>
  %w = stablehlo.constant dense_resource<w_bf16> : tensor<2x4xbf16>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<-0.5> : tensor<2x1xf32>}, decomposition = @fq_impl_bf16} : (tensor<2x4xbf16>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

{-#
  dialect_resources: {
    builtin: {
      w_bf16: "0x0200000040BF80BE803E403FC03E003E00BEC0BE"
    }
  }
#-}
