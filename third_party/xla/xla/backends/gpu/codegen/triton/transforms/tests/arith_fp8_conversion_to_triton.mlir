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
// RUN: xla-opt %s -arith-fp8-conversion-to-triton | FileCheck %s

// CHECK-LABEL: @extf_fp8_to_bf16
func.func @extf_fp8_to_bf16(%arg0: tensor<16xf8E5M2>) -> tensor<16xbf16> {
  // CHECK: tt.fp_to_fp
  %0 = arith.extf %arg0 : tensor<16xf8E5M2> to tensor<16xbf16>
  return %0 : tensor<16xbf16>
}

// CHECK-LABEL: @extf_fp8_to_f32
func.func @extf_fp8_to_f32(%arg0: tensor<32xf8E4M3FN>) -> tensor<32xf32> {
  // CHECK: tt.fp_to_fp
  %0 = arith.extf %arg0 : tensor<32xf8E4M3FN> to tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: @truncf_bf16_to_fp8e5m2_round_to_nearest_even
func.func @truncf_bf16_to_fp8e5m2_round_to_nearest_even(%arg0: tensor<16xbf16>) -> tensor<16xf8E5M2> {
  // CHECK: tt.fp_to_fp %{{.*}}, rounding = rtne
  %0 = arith.truncf %arg0 to_nearest_even : tensor<16xbf16> to tensor<16xf8E5M2>
  return %0 : tensor<16xf8E5M2>
}

// CHECK-LABEL: @truncf_f32_to_fp8e4m3fn_round_to_zero
func.func @truncf_f32_to_fp8e4m3fn_round_to_zero(%arg0: tensor<32xf32>) -> tensor<32xf8E4M3FN> {
  // CHECK: tt.fp_to_fp %{{.*}}, rounding = rtz
  %0 = arith.truncf %arg0 toward_zero : tensor<32xf32> to tensor<32xf8E4M3FN>
  return %0 : tensor<32xf8E4M3FN>
}

// CHECK-LABEL: @truncf_f32_to_fp8e4m3fn_no_rounding_mode_uses_nearest_even
func.func @truncf_f32_to_fp8e4m3fn_no_rounding_mode_uses_nearest_even(%arg0: tensor<32xf32>) -> tensor<32xf8E4M3FN> {
  // CHECK: tt.fp_to_fp %{{.*}}, rounding = rtne
  %0 = arith.truncf %arg0 : tensor<32xf32> to tensor<32xf8E4M3FN>
  return %0 : tensor<32xf8E4M3FN>
}

// CHECK-LABEL: @truncf_f32_to_fp8e4m3fn_unsupported_rounding_mode_falls_back_to_arith
func.func @truncf_f32_to_fp8e4m3fn_unsupported_rounding_mode_falls_back_to_arith(%arg0: tensor<32xf32>) -> tensor<32xf8E4M3FN> {
  // CHECK: arith.truncf
  %0 = arith.truncf %arg0 upward : tensor<32xf32> to tensor<32xf8E4M3FN>
  return %0 : tensor<32xf8E4M3FN>
}

// CHECK-LABEL: @truncf_f32_to_f16_falls_back_to_arith
func.func @truncf_f32_to_f16_falls_back_to_arith(%arg0: tensor<32xf32>) -> tensor<32xf16> {
  // CHECK: arith.truncf
  %0 = arith.truncf %arg0 to_nearest_even : tensor<32xf32> to tensor<32xf16>
  return %0 : tensor<32xf16>
}

// CHECK-LABEL: @extf_f16_to_f64_falls_back_to_arith
func.func @extf_f16_to_f64_falls_back_to_arith(%arg0: tensor<32xf16>) -> tensor<32xf64> {
  // CHECK: arith.extf
  %0 = arith.extf %arg0 : tensor<32xf16> to tensor<32xf64>
  return %0 : tensor<32xf64>
}
