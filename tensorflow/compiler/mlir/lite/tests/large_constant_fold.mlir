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

// RUN: litert-opt %s -tfl-large-constant-fold="fold-fp16-resource-casts=true" -split-input-file | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: litert-opt %s -tfl-large-constant-fold="fold-fp16-resource-casts=false" -split-input-file | FileCheck %s --check-prefixes=CHECK,NO_FP16_FOLD

// Mode 1: Exclusive transpose folds in place and updates the constant.
// CHECK-LABEL: func.func @test_transpose_in_place_exclusive
func.func @test_transpose_in_place_exclusive() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_t1> : tensor<3x2xf32>
  // CHECK: return %[[CST]] : tensor<3x2xf32>
  %0 = arith.constant dense_resource<res_t1> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_t1: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Mode 1: Exclusive cast folds in place and updates the constant.
// CHECK-LABEL: func.func @test_cast_in_place_exclusive
func.func @test_cast_in_place_exclusive() -> tensor<2x2xf32> {
  // DEFAULT: %[[CST:.*]] = arith.constant dense_resource<res_c1_cast_f32> : tensor<2x2xf32>
  // DEFAULT: return %[[CST]] : tensor<2x2xf32>
  // NO_FP16_FOLD: %[[CST:.*]] = arith.constant dense_resource<res_c1> : tensor<2x2xbf16>
  // NO_FP16_FOLD: %[[CAST:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // NO_FP16_FOLD: return %[[CAST]] : tensor<2x2xf32>
  %0 = arith.constant dense_resource<res_c1> : tensor<2x2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_c1: "0x40000000003F004000400040"
    }
  }
#-}

// -----

// Mode 1: Multiple candidate users with identical parameters reuse the single in-place folded constant.
// CHECK-LABEL: func.func @test_transpose_multiple_candidates_same_perm
func.func @test_transpose_multiple_candidates_same_perm() -> (tensor<3x2xf32>, tensor<3x2xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_t_multi> : tensor<3x2xf32>
  // CHECK: return %[[CST]], %[[CST]] : tensor<3x2xf32>, tensor<3x2xf32>
  %0 = arith.constant dense_resource<res_t_multi> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %2 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1, %2 : tensor<3x2xf32>, tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_t_multi: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Mode 2: Multiple candidate users with different parameters replicate out-of-place.
// Original constant is removed because all users were candidates.
// CHECK-LABEL: func.func @test_transpose_multiple_candidates_different_perms
func.func @test_transpose_multiple_candidates_different_perms() -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  // CHECK-NOT: arith.constant dense_resource<res_t_diff> :
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense_resource<res_t_diff_transpose_1_0> : tensor<3x2xf32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense_resource<res_t_diff_transpose_0_1> : tensor<2x3xf32>
  // CHECK: return %[[CST1]], %[[CST2]] : tensor<3x2xf32>, tensor<2x3xf32>
  %0 = arith.constant dense_resource<res_t_diff> : tensor<2x3xf32>
  %perm1 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %perm2 = arith.constant dense<[0, 1]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %2 = "tfl.transpose"(%0, %perm2) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  return %1, %2 : tensor<3x2xf32>, tensor<2x3xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_t_diff: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Mode 2: Multiple cast users with different target types replicate out-of-place.
// CHECK-LABEL: func.func @test_cast_multiple_candidates_different_types
func.func @test_cast_multiple_candidates_different_types() -> (tensor<2x2xf32>, tensor<2x2xf16>) {
  // DEFAULT-NOT: arith.constant dense_resource<res_c_diff> :
  // DEFAULT-DAG: %[[CST_F32:.*]] = arith.constant dense_resource<res_c_diff_cast_f32> : tensor<2x2xf32>
  // DEFAULT-DAG: %[[CST_F16:.*]] = arith.constant dense_resource<res_c_diff_cast_f16> : tensor<2x2xf16>
  // DEFAULT: return %[[CST_F32]], %[[CST_F16]] : tensor<2x2xf32>, tensor<2x2xf16>
  // NO_FP16_FOLD: %[[CST:.*]] = arith.constant dense_resource<res_c_diff> : tensor<2x2xbf16>
  // NO_FP16_FOLD-DAG: %[[CAST1:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // NO_FP16_FOLD-DAG: %[[CAST2:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2x2xbf16>) -> tensor<2x2xf16>
  // NO_FP16_FOLD: return %[[CAST1]], %[[CAST2]] : tensor<2x2xf32>, tensor<2x2xf16>
  %0 = arith.constant dense_resource<res_c_diff> : tensor<2x2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  %2 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf16>
  return %1, %2 : tensor<2x2xf32>, tensor<2x2xf16>
}

{-#
  dialect_resources: {
    builtin: {
      res_c_diff: "0x40000000003F004000400040"
    }
  }
#-}

// -----

// Mode 2: Constant has a candidate consumer and a non-candidate consumer.
// Original constant is preserved untouched for the non-candidate consumer.
// CHECK-LABEL: func.func @test_transpose_with_non_candidate_consumer
func.func @test_transpose_with_non_candidate_consumer() -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  // CHECK-DAG: %[[ORIG:.*]] = arith.constant dense_resource<res_non_cand> : tensor<2x3xf32>
  // CHECK-DAG: %[[TRANS:.*]] = arith.constant dense_resource<res_non_cand_transpose_1_0> : tensor<3x2xf32>
  // CHECK: return %[[TRANS]], %[[ORIG]] : tensor<3x2xf32>, tensor<2x3xf32>
  %0 = arith.constant dense_resource<res_non_cand> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1, %0 : tensor<3x2xf32>, tensor<2x3xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_non_cand: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Mode 2: Constant has a cast consumer and a non-candidate consumer.
// Original constant is preserved untouched for the non-candidate consumer.
// CHECK-LABEL: func.func @test_cast_with_non_candidate_consumer
func.func @test_cast_with_non_candidate_consumer() -> (tensor<2x2xf32>, tensor<2x2xbf16>) {
  // DEFAULT-DAG: %[[ORIG:.*]] = arith.constant dense_resource<res_cast_non_cand> : tensor<2x2xbf16>
  // DEFAULT-DAG: %[[CAST:.*]] = arith.constant dense_resource<res_cast_non_cand_cast_f32> : tensor<2x2xf32>
  // DEFAULT: return %[[CAST]], %[[ORIG]] : tensor<2x2xf32>, tensor<2x2xbf16>
  // NO_FP16_FOLD: %[[ORIG:.*]] = arith.constant dense_resource<res_cast_non_cand> : tensor<2x2xbf16>
  // NO_FP16_FOLD: %[[CAST:.*]] = "tfl.cast"(%[[ORIG]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // NO_FP16_FOLD: return %[[CAST]], %[[ORIG]] : tensor<2x2xf32>, tensor<2x2xbf16>
  %0 = arith.constant dense_resource<res_cast_non_cand> : tensor<2x2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  return %1, %0 : tensor<2x2xf32>, tensor<2x2xbf16>
}

{-#
  dialect_resources: {
    builtin: {
      res_cast_non_cand: "0x40000000003F004000400040"
    }
  }
#-}

// -----

// Mode 2: Two distinct constants share the same resource blob across functions.
// One constant is used by a candidate, the other by a non-candidate.
// The candidate constant replicates out-of-place; the non-candidate constant is preserved.
// CHECK-LABEL: func.func @func_with_transpose
// CHECK: %[[TRANS:.*]] = arith.constant dense_resource<res_shared_transpose_1_0> : tensor<3x2xf32>
// CHECK: return %[[TRANS]] : tensor<3x2xf32>
func.func @func_with_transpose() -> tensor<3x2xf32> {
  %0 = arith.constant dense_resource<res_shared> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @func_with_direct_use
// CHECK: %[[ORIG:.*]] = arith.constant dense_resource<res_shared> : tensor<2x3xf32>
// CHECK: return %[[ORIG]] : tensor<2x3xf32>
func.func @func_with_direct_use() -> tensor<2x3xf32> {
  %0 = arith.constant dense_resource<res_shared> : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_shared: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Mode 1: Transpose with i64 permutation tensor.
// CHECK-LABEL: func.func @test_i64_perm_transpose
func.func @test_i64_perm_transpose() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_i64_perm> : tensor<3x2xf32>
  // CHECK: return %[[CST]] : tensor<3x2xf32>
  %0 = arith.constant dense_resource<res_i64_perm> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi64>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_i64_perm: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Mode 1: Exclusive sub-byte (4-bit) transpose folds in place.
// CHECK-LABEL: func.func @test_transpose_subbyte_4bit
func.func @test_transpose_subbyte_4bit() -> tensor<4x2xi4> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_i4_2x4> : tensor<4x2xi4>
  // CHECK: return %[[CST]] : tensor<4x2xi4>
  %0 = arith.constant dense_resource<res_i4_2x4> : tensor<2x4xi4>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x4xi4>, tensor<2xi32>) -> tensor<4x2xi4>
  return %1 : tensor<4x2xi4>
}

{-#
  dialect_resources: {
    builtin: {
      res_i4_2x4: "0x0400000012345678"
    }
  }
#-}

// -----

// Unsupported bit-width (5-bit) transpose folding fails and preserves transpose op.
// CHECK-LABEL: func.func @test_transpose_unsupported_bitwidth_not_folded
func.func @test_transpose_unsupported_bitwidth_not_folded() -> tensor<3x2xi5> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_i5_2x3> : tensor<2x3xi5>
  // CHECK: %[[PERM:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %[[TRANS:.*]] = "tfl.transpose"(%[[CST]], %[[PERM]]) : (tensor<2x3xi5>, tensor<2xi32>) -> tensor<3x2xi5>
  // CHECK: return %[[TRANS]] : tensor<3x2xi5>
  %0 = arith.constant dense_resource<res_i5_2x3> : tensor<2x3xi5>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xi5>, tensor<2xi32>) -> tensor<3x2xi5>
  return %1 : tensor<3x2xi5>
}

{-#
  dialect_resources: {
    builtin: {
      res_i5_2x3: "0x0400000011223344"
    }
  }
#-}



