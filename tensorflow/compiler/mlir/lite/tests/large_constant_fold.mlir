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

// RUN: litert-opt %s -tfl-large-constant-fold -split-input-file | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: litert-opt %s -tfl-large-constant-fold="fold-elementwise-ops=true" -split-input-file | FileCheck %s --check-prefixes=CHECK,ELEMWISE
// RUN: litert-opt %s -tfl-large-constant-fold="fold-fp16-resource-casts=false" -split-input-file | FileCheck %s --check-prefixes=CHECK,NO_FP16_FOLD

// Tests folding of tfl.cast when input and output element types are identical (f32 -> f32).
// Verifies that tfl.cast is replaced with arith.constant referencing the original resource handle.
// CHECK-LABEL: func.func @test_identity_resource_cast
func.func @test_identity_resource_cast() -> tensor<4xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_f32_identity> : tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = arith.constant dense_resource<res_f32_identity> : tensor<4xf32>
  %1 = "tfl.cast"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_f32_identity: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests folding of tfl.cast on a non-splat bf16 resource constant converting to f32.
// Verifies that litert.target_element_type = f32 attribute is set when fold-fp16-resource-casts=true, and cast is retained when fold-fp16-resource-casts=false.
// CHECK-LABEL: func.func @test_non_splat_resource_cast
func.func @test_non_splat_resource_cast() -> tensor<2x2xf32> {
  // DEFAULT: %[[CST:.*]] = arith.constant {litert.target_element_type = f32} dense_resource<res_bf16_nonsplat> : tensor<2x2xf32>
  // DEFAULT-NEXT: return %[[CST]] : tensor<2x2xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant {litert.target_element_type = f32} dense_resource<res_bf16_nonsplat> : tensor<2x2xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<2x2xf32>
  // NO_FP16_FOLD: %[[CST:.*]] = arith.constant dense_resource<res_bf16_nonsplat> : tensor<2x2xbf16>
  // NO_FP16_FOLD: %[[CAST:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // NO_FP16_FOLD-NEXT: return %[[CAST]] : tensor<2x2xf32>
  %0 = arith.constant dense_resource<res_bf16_nonsplat> : tensor<2x2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_bf16_nonsplat: "0x40000000003F004000400040"
    }
  }
#-}

// -----

// Tests folding of tfl.cast on a splat bf16 resource constant converting to f32.
// Verifies that a new resource constant with cast_..._f32 handle is generated when fold-fp16-resource-casts=true, and cast is retained when fold-fp16-resource-casts=false.
// CHECK-LABEL: func.func @test_splat_resource_cast
func.func @test_splat_resource_cast() -> tensor<2x2xf32> {
  // DEFAULT: %[[CST:.*]] = arith.constant dense_resource<{{cast_.*_f32}}> : tensor<2x2xf32>
  // DEFAULT-NEXT: return %[[CST]] : tensor<2x2xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant dense_resource<{{cast_.*_f32}}> : tensor<2x2xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<2x2xf32>
  // NO_FP16_FOLD: %[[CST:.*]] = arith.constant dense_resource<res_bf16_splat> : tensor<2x2xbf16>
  // NO_FP16_FOLD: %[[CAST:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // NO_FP16_FOLD-NEXT: return %[[CAST]] : tensor<2x2xf32>
  %0 = arith.constant dense_resource<res_bf16_splat> : tensor<2x2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_bf16_splat: "0x40000000003F"
    }
  }
#-}

// -----

// Tests folding of tfl.transpose on a non-splat 2D resource constant.
// Verifies that the constant's result type and value attribute type are updated to transposed shape (tensor<3x2xf32>) and litert.layout_permutation = [1, 0] is set.
// CHECK-LABEL: func.func @test_non_splat_resource_transpose
func.func @test_non_splat_resource_transpose() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_f32_2x3> : tensor<3x2xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<3x2xf32>
  %0 = arith.constant dense_resource<res_f32_2x3> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_f32_2x3: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Tests folding of tfl.transpose on a 1D resource constant.
// Verifies that the transpose is eliminated and the constant shape is preserved.
// CHECK-LABEL: func.func @test_1d_resource_transpose
func.func @test_1d_resource_transpose() -> tensor<4xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_f32_1d> : tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = arith.constant dense_resource<res_f32_1d> : tensor<4xf32>
  %perm = arith.constant dense<[0]> : tensor<1xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<4xf32>, tensor<1xi32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_f32_1d: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests folding of tfl.reshape on a resource constant.
// Verifies that tfl.reshape is replaced with an arith.constant of target shape (tensor<3x2xf32>) reusing the original resource handle.
// CHECK-LABEL: func.func @test_resource_reshape
func.func @test_resource_reshape() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<{{res_f32_reshape.*}}> : tensor<3x2xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<3x2xf32>
  %0 = arith.constant dense_resource<res_f32_reshape> : tensor<2x3xf32>
  %shape = arith.constant dense<[3, 2]> : tensor<2xi32>
  %1 = "tfl.reshape"(%0, %shape) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_f32_reshape: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Tests folding of tfl.cast on standard dense attributes from bf16 to f32.
// Verifies that tfl.cast is replaced by an arith.constant containing converted dense f32 elements.
// CHECK-LABEL: func.func @test_dense_elements_cast
func.func @test_dense_elements_cast() -> tensor<2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<2xf32>
  %0 = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2xbf16>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----

// Tests tfl.add folding behavior (retained by default, folded when fold-elementwise-ops=true).
// CHECK-LABEL: func.func @test_resource_add
func.func @test_resource_add() -> tensor<4xf32> {
  // DEFAULT: %[[LHS:.*]] = arith.constant dense_resource<res_add_lhs> : tensor<4xf32>
  // DEFAULT: %[[RHS:.*]] = arith.constant dense_resource<res_add_rhs> : tensor<4xf32>
  // DEFAULT: %[[RES:.*]] = tfl.add %[[LHS]], %[[RHS]] {fused_activation_function = "NONE"} : tensor<4xf32>
  // DEFAULT-NEXT: return %[[RES]] : tensor<4xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant dense_resource<{{add_.*}}> : tensor<4xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = arith.constant dense_resource<res_add_lhs> : tensor<4xf32>
  %1 = arith.constant dense_resource<res_add_rhs> : tensor<4xf32>
  %2 = "tfl.add"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_add_lhs: "0x400000000000803F000000400000404000008040",
      res_add_rhs: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests tfl.sub folding behavior (retained by default, folded when fold-elementwise-ops=true).
// CHECK-LABEL: func.func @test_resource_sub
func.func @test_resource_sub() -> tensor<4xf32> {
  // DEFAULT: %[[LHS:.*]] = arith.constant dense_resource<res_sub_lhs> : tensor<4xf32>
  // DEFAULT: %[[RHS:.*]] = arith.constant dense_resource<res_sub_rhs> : tensor<4xf32>
  // DEFAULT: %[[RES:.*]] = tfl.sub %[[LHS]], %[[RHS]] {fused_activation_function = "NONE"} : tensor<4xf32>
  // DEFAULT-NEXT: return %[[RES]] : tensor<4xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant dense_resource<{{sub_.*}}> : tensor<4xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = arith.constant dense_resource<res_sub_lhs> : tensor<4xf32>
  %1 = arith.constant dense_resource<res_sub_rhs> : tensor<4xf32>
  %2 = "tfl.sub"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_sub_lhs: "0x400000000000803F000000400000404000008040",
      res_sub_rhs: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests tfl.mul folding behavior (retained by default, folded when fold-elementwise-ops=true).
// CHECK-LABEL: func.func @test_resource_mul
func.func @test_resource_mul() -> tensor<4xf32> {
  // DEFAULT: %[[LHS:.*]] = arith.constant dense_resource<res_mul_lhs> : tensor<4xf32>
  // DEFAULT: %[[RHS:.*]] = arith.constant dense_resource<res_mul_rhs> : tensor<4xf32>
  // DEFAULT: %[[RES:.*]] = tfl.mul %[[LHS]], %[[RHS]] {fused_activation_function = "NONE"} : tensor<4xf32>
  // DEFAULT-NEXT: return %[[RES]] : tensor<4xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant dense_resource<{{mul_.*}}> : tensor<4xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = arith.constant dense_resource<res_mul_lhs> : tensor<4xf32>
  %1 = arith.constant dense_resource<res_mul_rhs> : tensor<4xf32>
  %2 = "tfl.mul"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_mul_lhs: "0x400000000000803F000000400000404000008040",
      res_mul_rhs: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests tfl.div folding behavior (retained by default, folded when fold-elementwise-ops=true).
// CHECK-LABEL: func.func @test_resource_div
func.func @test_resource_div() -> tensor<4xf32> {
  // DEFAULT: %[[LHS:.*]] = arith.constant dense_resource<res_div_lhs> : tensor<4xf32>
  // DEFAULT: %[[RHS:.*]] = arith.constant dense_resource<res_div_rhs> : tensor<4xf32>
  // DEFAULT: %[[RES:.*]] = tfl.div %[[LHS]], %[[RHS]] {fused_activation_function = "NONE"} : tensor<4xf32>
  // DEFAULT-NEXT: return %[[RES]] : tensor<4xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant dense_resource<{{div_.*}}> : tensor<4xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<4xf32>
  %0 = arith.constant dense_resource<res_div_lhs> : tensor<4xf32>
  %1 = arith.constant dense_resource<res_div_rhs> : tensor<4xf32>
  %2 = "tfl.div"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_div_lhs: "0x400000000000803F000000400000404000008040",
      res_div_rhs: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests folding of tfl.transpose on a resource constant with an existing litert.layout_permutation attribute.
// Verifies that old permutation [1, 0, 2] and new permutation [2, 0, 1] are composed into [2, 1, 0].
// CHECK-LABEL: func.func @test_composed_layout_permutation_transpose
func.func @test_composed_layout_permutation_transpose() -> tensor<2x2x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant {litert.layout_permutation = [2, 1, 0]} dense_resource<res_3d> : tensor<2x2x2xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x2x2xf32>
  %0 = arith.constant {litert.layout_permutation = [1, 0, 2]} dense_resource<res_3d> : tensor<2x2x2xf32>
  %perm = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
  return %1 : tensor<2x2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_3d: "0x400000000000803F0000004000004040000080400000A0400000C0400000E04000000041"
    }
  }
#-}

// -----

// Tests folding of tfl.cast on a resource constant with an existing litert.layout_permutation attribute.
// Verifies that litert.layout_permutation attribute is preserved and forwarded onto the new constant op when fold-fp16-resource-casts=true, and cast is retained when fold-fp16-resource-casts=false.
// CHECK-LABEL: func.func @test_cast_forward_layout_permutation
func.func @test_cast_forward_layout_permutation() -> tensor<2x2xf32> {
  // DEFAULT: %[[CST:.*]] = arith.constant {litert.layout_permutation = [1, 0], litert.target_element_type = f32} dense_resource<res_perm_bf16> : tensor<2x2xf32>
  // DEFAULT-NEXT: return %[[CST]] : tensor<2x2xf32>
  // ELEMWISE: %[[CST:.*]] = arith.constant {litert.layout_permutation = [1, 0], litert.target_element_type = f32} dense_resource<res_perm_bf16> : tensor<2x2xf32>
  // ELEMWISE-NEXT: return %[[CST]] : tensor<2x2xf32>
  // NO_FP16_FOLD: %[[CST:.*]] = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_perm_bf16> : tensor<2x2xbf16>
  // NO_FP16_FOLD: %[[CAST:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  // NO_FP16_FOLD-NEXT: return %[[CAST]] : tensor<2x2xf32>
  %0 = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_perm_bf16> : tensor<2x2xbf16>
  %1 = "tfl.cast"(%0) : (tensor<2x2xbf16>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_perm_bf16: "0x40000000003F004000400040"
    }
  }
#-}

// -----

// Tests folding of tfl.transpose where the permutation tensor has i64 element type.
// Verifies that i64 permutations are parsed correctly and set litert.layout_permutation = [1, 0].
// CHECK-LABEL: func.func @test_i64_perm_transpose
func.func @test_i64_perm_transpose() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_f32_i64_perm> : tensor<3x2xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<3x2xf32>
  %0 = arith.constant dense_resource<res_f32_i64_perm> : tensor<2x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi64>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_f32_i64_perm: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Tests folding of tfl.cast on a splat f32 resource constant converting to bf16.
// Verifies that a new bf16 resource constant with cast_..._bf16 handle is generated.
// CHECK-LABEL: func.func @test_cast_f32_to_bf16
func.func @test_cast_f32_to_bf16() -> tensor<2x2xbf16> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<{{cast_.*_bf16}}> : tensor<2x2xbf16>
  // CHECK-NEXT: return %[[CST]] : tensor<2x2xbf16>
  %0 = arith.constant dense_resource<res_f32_splat> : tensor<2x2xf32>
  %1 = "tfl.cast"(%0) : (tensor<2x2xf32>) -> tensor<2x2xbf16>
  return %1 : tensor<2x2xbf16>
}

{-#
  dialect_resources: {
    builtin: {
      res_f32_splat: "0x400000000000803F"
    }
  }
#-}

// -----

// Tests folding of binary op (tfl.add) when one operand is a resource constant
// and the other operand is a regular dense constant.
// Verifies that different dense constant operands of the same shape and size do
// not cause cache collisions resulting in incorrect folded values.
// CHECK-LABEL: func.func @test_dense_operand_cache_collision
func.func @test_dense_operand_cache_collision() -> (tensor<4xf32>, tensor<4xf32>) {
  // DEFAULT: %[[RES:.*]] = arith.constant dense_resource<res_add_dense_collision_lhs> : tensor<4xf32>
  // DEFAULT-NEXT: %[[CST1:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // DEFAULT-NEXT: %[[CST2:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf32>
  // DEFAULT-NEXT: %[[ADD1:.*]] = tfl.add %[[RES]], %[[CST1]] {fused_activation_function = "NONE"} : tensor<4xf32>
  // DEFAULT-NEXT: %[[ADD2:.*]] = tfl.add %[[RES]], %[[CST2]] {fused_activation_function = "NONE"} : tensor<4xf32>
  // DEFAULT-NEXT: return %[[ADD1]], %[[ADD2]] : tensor<4xf32>, tensor<4xf32>
  // ELEMWISE: %[[RES1:.*]] = arith.constant dense_resource<{{add_.*}}> : tensor<4xf32>
  // ELEMWISE-NEXT: %[[RES2:.*]] = arith.constant dense_resource<{{add_.*}}> : tensor<4xf32>
  // ELEMWISE-NEXT: return %[[RES1]], %[[RES2]] : tensor<4xf32>, tensor<4xf32>
  %0 = arith.constant dense_resource<res_add_dense_collision_lhs> : tensor<4xf32>
  %1 = arith.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %2 = arith.constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
  %3 = "tfl.add"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "tfl.add"(%0, %2) <{fused_activation_function = "NONE"}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3, %4 : tensor<4xf32>, tensor<4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_add_dense_collision_lhs: "0x400000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Tests folding of tfl.div with zero divisor on integer types.
// Verifies that division by zero is skipped to avoid crash/UB.
// CHECK-LABEL: func.func @test_resource_div_by_zero
func.func @test_resource_div_by_zero() -> tensor<4xi32> {
  // CHECK: %[[LHS:.*]] = arith.constant dense_resource<res_div_zero_lhs> : tensor<4xi32>
  // CHECK-NEXT: %[[RHS:.*]] = arith.constant dense<0> : tensor<4xi32>
  // CHECK-NEXT: %[[RES:.*]] = tfl.div %[[LHS]], %[[RHS]] {fused_activation_function = "NONE"} : tensor<4xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<4xi32>
  %0 = arith.constant dense_resource<res_div_zero_lhs> : tensor<4xi32>
  %1 = arith.constant dense<0> : tensor<4xi32>
  %2 = "tfl.div"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %2 : tensor<4xi32>
}

{-#
  dialect_resources: {
    builtin: {
      res_div_zero_lhs: "0x01000000020000000300000004000000"
    }
  }
#-}

// -----

// Tests elementwise binary folding (tfl.add) when an input operand has a litert.layout_permutation attribute.
// Verifies that binary folding is skipped when an input has layout permutation.
// CHECK-LABEL: func.func @test_permuted_operand_binary_fold
func.func @test_permuted_operand_binary_fold() -> tensor<3x2xf32> {
  // ELEMWISE: %[[LHS:.*]] = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_permuted_lhs> : tensor<3x2xf32>
  // ELEMWISE-NEXT: %[[RHS:.*]] = arith.constant dense_resource<res_permuted_rhs> : tensor<3x2xf32>
  // ELEMWISE-NEXT: %[[RES:.*]] = tfl.add %[[LHS]], %[[RHS]] {fused_activation_function = "NONE"} : tensor<3x2xf32>
  // ELEMWISE-NEXT: return %[[RES]] : tensor<3x2xf32>
  %0 = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_permuted_lhs> : tensor<3x2xf32>
  %1 = arith.constant dense_resource<res_permuted_rhs> : tensor<3x2xf32>
  %2 = "tfl.add"(%0, %1) <{fused_activation_function = "NONE"}> : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %2 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_permuted_lhs: "0x400000000000803F0000004000004040000080400000A0400000C040",
      res_permuted_rhs: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}

// -----

// Tests folding of tfl.reshape on a resource constant when the input has a litert.layout_permutation attribute.
// Verifies that reshape folding is skipped (returns failure) to avoid dropping the layout permutation.
// CHECK-LABEL: func.func @test_permuted_resource_reshape
func.func @test_permuted_resource_reshape() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_permuted_reshape> : tensor<2x3xf32>
  // CHECK-NEXT: %[[SHAPE:.*]] = arith.constant dense<[3, 2]> : tensor<2xi32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = "tfl.reshape"(%[[CST]], %[[SHAPE]]) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<3x2xf32>
  %0 = arith.constant {litert.layout_permutation = [1, 0]} dense_resource<res_permuted_reshape> : tensor<2x3xf32>
  %shape = arith.constant dense<[3, 2]> : tensor<2xi32>
  %1 = "tfl.reshape"(%0, %shape) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_permuted_reshape: "0x400000000000803F0000004000004040000080400000A0400000C040"
    }
  }
#-}


