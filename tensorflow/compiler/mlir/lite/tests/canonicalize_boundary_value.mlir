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
// RUN: litert-opt %s --tfl-canonicalize-boundary-value --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @clamp_neg_inf_f32() -> tensor<f32> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           return %[[CONST]] : tensor<f32>

func.func @clamp_neg_inf_f32() -> tensor<f32> {
  %ret = stablehlo.constant dense<0xFF800000> : tensor<f32>
  return %ret : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f32() -> tensor<f32> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           return %[[CONST]] : tensor<f32>
func.func @clamp_pos_inf_f32() -> tensor<f32> {
  %ret = stablehlo.constant dense<0x7F800000> : tensor<f32>
  return %ret : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f32_tensor() -> tensor<1x4xf32> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<{{\[\[}}3.40282347E+38, 1.000000e+01, 2.000000e+01, -3.40282347E+38]]> : tensor<1x4xf32>
// CHECK:           return %[[CONST]] : tensor<1x4xf32>
func.func @clamp_pos_inf_f32_tensor() -> tensor<1x4xf32> {
  %ret = stablehlo.constant dense<[[0x7F800000, 10.0, 20.0, 0xFF800000]]> : tensor<1x4xf32>
  return %ret : tensor<1x4xf32>
}

// -----

// CHECK-LABEL:   func.func @clamp_neg_inf_f16() -> tensor<f16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
// CHECK:           return %[[CONST]] : tensor<f16>
func.func @clamp_neg_inf_f16() -> tensor<f16> {
  %ret = stablehlo.constant dense<0xFC00> : tensor<f16>
  return %ret : tensor<f16>
}
// -----

// CHECK-LABEL:   func.func @clamp_neg_inf_bf16() -> tensor<bf16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
// CHECK:           return %[[CONST]] : tensor<bf16>
func.func @clamp_neg_inf_bf16() -> tensor<bf16> {
  %ret = stablehlo.constant dense<0xFF80> : tensor<bf16>
  return %ret : tensor<bf16>
}

// -----

// CHECK-LABEL:   func.func @clamp_inf_bf16_tensor() -> tensor<3xbf16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<[3.389530e+38, 1.000000e+00, -3.389530e+38]> : tensor<3xbf16>
// CHECK:           return %[[CONST]] : tensor<3xbf16>
func.func @clamp_inf_bf16_tensor() -> tensor<3xbf16> {
  %ret = stablehlo.constant dense<[0x7F80, 1.0, 0xFF80]> : tensor<3xbf16>
  return %ret : tensor<3xbf16>
}

// -----

// CHECK-LABEL:   func.func @clamp_inf_f64_tensor() -> tensor<3xf64> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<[1.7976931348623157E+308, 1.000000e+00, -1.7976931348623157E+308]> : tensor<3xf64>
// CHECK:           return %[[CONST]] : tensor<3xf64>
func.func @clamp_inf_f64_tensor() -> tensor<3xf64> {
  %ret = stablehlo.constant dense<[0x7FF0000000000000, 1.0, 0xFFF0000000000000]> : tensor<3xf64>
  return %ret : tensor<3xf64>
}

// -----

// CHECK-LABEL:   func.func @no_clamp_finite_and_nan_tensors() -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf16>, tensor<2xbf16>) {
// CHECK:           %[[F32_FINITE:.*]] = stablehlo.constant dense<[1.000000e+00, -2.000000e+00]> : tensor<2xf32>
// CHECK:           %[[F32_NAN:.*]] = stablehlo.constant dense<[1.000000e+00, 0x7FC00000]> : tensor<2xf32>
// CHECK:           %[[F16_NAN:.*]] = stablehlo.constant dense<[1.000000e+00, 0x7E00]> : tensor<2xf16>
// CHECK:           %[[BF16_NAN:.*]] = stablehlo.constant dense<[1.000000e+00, 0x7FC0]> : tensor<2xbf16>
// CHECK:           return %[[F32_FINITE]], %[[F32_NAN]], %[[F16_NAN]], %[[BF16_NAN]]
func.func @no_clamp_finite_and_nan_tensors() -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf16>, tensor<2xbf16>) {
  %0 = stablehlo.constant dense<[1.0, -2.0]> : tensor<2xf32>
  %1 = stablehlo.constant dense<[1.0, 0x7FC00000]> : tensor<2xf32>
  %2 = stablehlo.constant dense<[1.0, 0x7E00]> : tensor<2xf16>
  %3 = stablehlo.constant dense<[1.0, 0x7FC0]> : tensor<2xbf16>
  return %0, %1, %2, %3 : tensor<2xf32>, tensor<2xf32>, tensor<2xf16>, tensor<2xbf16>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16() -> tensor<f16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<6.550400e+04> : tensor<f16>
// CHECK:           return %[[CONST]] : tensor<f16>
func.func @clamp_pos_inf_f16() -> tensor<f16> {
  %ret = stablehlo.constant dense<0x7C00> : tensor<f16>
  return %ret : tensor<f16>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16_tensor() -> tensor<1x4xf16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<{{\[\[}}6.550400e+04, 1.000000e+01, 2.000000e+01, -6.550400e+04]]> : tensor<1x4xf16>
// CHECK:           return %[[CONST]] : tensor<1x4xf16>
func.func @clamp_pos_inf_f16_tensor() -> tensor<1x4xf16> {
  %ret = stablehlo.constant dense<[[0x7C00, 10.0, 20.0, 0xFC00]]> : tensor<1x4xf16>
  return %ret : tensor<1x4xf16>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16_tensor_tf_const() -> tensor<3xf16> {
// CHECK:           %[[CONST:.*]] = "tf.Const"() <{value = dense<6.550400e+04> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK:           return %[[CONST]] : tensor<3xf16>
func.func @clamp_pos_inf_f16_tensor_tf_const() -> tensor<3xf16> {
  %ret = "tf.Const"() <{value = dense<0x7C00> : tensor<3xf16>}> : () -> tensor<3xf16>
  return %ret : tensor<3xf16>
}


// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16_arith_op() -> f16 {
// CHECK:           %[[CONST:.*]] = arith.constant 6.550400e+04 : f16
// CHECK:           return %[[CONST]] : f16
func.func @clamp_pos_inf_f16_arith_op() -> f16 {
  %ret = arith.constant 0x7C00 : f16
  return %ret : f16
}

