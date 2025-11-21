// RUN: xla-opt %s -arith-fp8-conversion-to-triton | FileCheck %s

// CHECK: @extf_fp8_to_bf16(%[[ARG0:.*]]: tensor<16xf8E5M2>) -> tensor<16xbf16>
func.func @extf_fp8_to_bf16(%arg0: tensor<16xf8E5M2>) -> tensor<16xbf16> {
  // CHECK: tt.fp_to_fp %[[ARG0]] : tensor<16xf8E5M2> -> tensor<16xbf16>
  %0 = arith.extf %arg0 : tensor<16xf8E5M2> to tensor<16xbf16>
  return %0 : tensor<16xbf16>
}

// CHECK: @extf_fp8_to_f32(%[[ARG0:.*]]: tensor<32xf8E4M3FN>) -> tensor<32xf32>
func.func @extf_fp8_to_f32(%arg0: tensor<32xf8E4M3FN>) -> tensor<32xf32> {
  // CHECK: tt.fp_to_fp %[[ARG0]] : tensor<32xf8E4M3FN> -> tensor<32xf32>
  %0 = arith.extf %arg0 : tensor<32xf8E4M3FN> to tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK: @truncf_bf16_to_fp8e5m2_round_to_nearest_even(%[[ARG0:.*]]: tensor<16xbf16>) -> tensor<16xf8E5M2>
func.func @truncf_bf16_to_fp8e5m2_round_to_nearest_even(%arg0: tensor<16xbf16>) -> tensor<16xf8E5M2> {
  // CHECK: tt.fp_to_fp %[[ARG0]], rounding = rtne : tensor<16xbf16> -> tensor<16xf8E5M2>
  %0 = arith.truncf %arg0 to_nearest_even : tensor<16xbf16> to tensor<16xf8E5M2>
  return %0 : tensor<16xf8E5M2>
}

// CHECK: @truncf_f32_to_fp8e4m3fn_round_to_zero(%[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf8E4M3FN>
func.func @truncf_f32_to_fp8e4m3fn_round_to_zero(%arg0: tensor<32xf32>) -> tensor<32xf8E4M3FN> {
  // CHECK: tt.fp_to_fp %[[ARG0]], rounding = rtz : tensor<32xf32> -> tensor<32xf8E4M3FN>
  %0 = arith.truncf %arg0 toward_zero : tensor<32xf32> to tensor<32xf8E4M3FN>
  return %0 : tensor<32xf8E4M3FN>
}

// CHECK: @truncf_f32_to_fp8e4m3fn_no_rounding_mode_uses_nearest_even(%[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf8E4M3FN>
func.func @truncf_f32_to_fp8e4m3fn_no_rounding_mode_uses_nearest_even(%arg0: tensor<32xf32>) -> tensor<32xf8E4M3FN> {
  // CHECK: tt.fp_to_fp %[[ARG0]], rounding = rtne : tensor<32xf32> -> tensor<32xf8E4M3FN>
  %0 = arith.truncf %arg0 : tensor<32xf32> to tensor<32xf8E4M3FN>
  return %0 : tensor<32xf8E4M3FN>
}

// CHECK: @truncf_f32_to_fp8e4m3fn_unsupported_rounding_mode_falls_back_to_arith(%[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf8E4M3FN>
func.func @truncf_f32_to_fp8e4m3fn_unsupported_rounding_mode_falls_back_to_arith(%arg0: tensor<32xf32>) -> tensor<32xf8E4M3FN> {
  // CHECK: arith.truncf %[[ARG0]] upward : tensor<32xf32> to tensor<32xf8E4M3FN>
  %0 = arith.truncf %arg0 upward : tensor<32xf32> to tensor<32xf8E4M3FN>
  return %0 : tensor<32xf8E4M3FN>
}

// CHECK: @truncf_f32_to_f16_falls_back_to_arith(%[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf16>
func.func @truncf_f32_to_f16_falls_back_to_arith(%arg0: tensor<32xf32>) -> tensor<32xf16> {
  // CHECK: arith.truncf %[[ARG0]] to_nearest_even : tensor<32xf32> to tensor<32xf16>
  %0 = arith.truncf %arg0 to_nearest_even : tensor<32xf32> to tensor<32xf16>
  return %0 : tensor<32xf16>
}

// CHECK: @extf_f16_to_f64_falls_back_to_arith(%[[ARG0:.*]]: tensor<32xf16>) -> tensor<32xf64>
func.func @extf_f16_to_f64_falls_back_to_arith(%arg0: tensor<32xf16>) -> tensor<32xf64> {
  // CHECK: arith.extf %[[ARG0]] : tensor<32xf16> to tensor<32xf64>
  %0 = arith.extf %arg0 : tensor<32xf16> to tensor<32xf64>
  return %0 : tensor<32xf64>
}
