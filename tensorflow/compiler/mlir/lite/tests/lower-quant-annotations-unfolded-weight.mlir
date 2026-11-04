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
// RUN: litert-opt %s --tfl-fold-stablehlo-constant-transforms --tfl-lower-quant-annotations | FileCheck %s
// RUN: litert-opt %s --tfl-lower-quant-annotations -verify-diagnostics

// A pre-quantized i4 weight that is transposed before it is dequantized. Once
// the transpose is folded into the constant, the dequantize lowers to a
// quantized constant. Without the fold, the pass reports the unfolded op
// instead of producing a transpose with mismatched element types.
// CHECK-LABEL: func.func @dequantize_transposed_i4_weight
func.func @dequantize_transposed_i4_weight() -> tensor<3x2xf32> {
  // CHECK: %[[QCST:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<3x2x!quant.uniform<i4:f32:1, {1.000000e+00,2.000000e+00}>>, value = dense_resource<packed_i4> : tensor<3x2xi4>}>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: %[[DQ:.+]] = "tfl.dequantize"(%[[QCST]])
  // CHECK: return %[[DQ]]
  %0 = stablehlo.constant dense_resource<packed_i4> : tensor<2x3xi4>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi4>) -> tensor<3x2xi4>
  // expected-error @+1 {{the input of quant.dequantize must be a constant, but it is produced by 'stablehlo.transpose' with operand type 'tensor<2x3xi4>', which was not folded into a constant}}
  %2 = stablehlo.custom_call @quant.dequantize(%1) {dtype = "i4", narrow_range = false, quantization_dimension = 1 : i32, scale = dense<[1.0, 2.0]> : tensor<2xf32>} : (tensor<3x2xi4>) -> tensor<3x2xf32>
  return %2 : tensor<3x2xf32>
}

// A shape op over a non-constant quantized value is still retyped.
// CHECK-LABEL: func.func @dequantize_reshaped_activation
func.func @dequantize_reshaped_activation(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
  // CHECK: %[[Q:.+]] = "tfl.quantize"(%arg0)
  // CHECK: %[[R:.+]] = stablehlo.reshape %[[Q]] : (tensor<2x2x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x!quant.uniform<i8:f32, 5.000000e-01>>
  // CHECK: %[[DQ:.+]] = "tfl.dequantize"(%[[R]])
  // CHECK: return %[[DQ]]
  %0 = stablehlo.custom_call @quant.quantize(%arg0) {dtype = "i8", narrow_range = false, scale = dense<0.5> : tensor<1xf32>} : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %1 = stablehlo.reshape %0 : (tensor<2x2xi8>) -> tensor<4xi8>
  %2 = stablehlo.custom_call @quant.dequantize(%1) {dtype = "i8", narrow_range = false, scale = dense<0.5> : tensor<1xf32>} : (tensor<4xi8>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      packed_i4: "0x04000000214365"
    }
  }
#-}
