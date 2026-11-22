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

// RUN: litert-opt %s -tfl-large-constant-fold="fold-elementwise-ops=true" -split-input-file | FileCheck %s
// RUN: litert-opt %s -tfl-large-constant-fold="fold-elementwise-ops=false" -split-input-file | FileCheck %s --check-prefix=DISABLED

// An exclusively owned resource is updated in place.
// CHECK-LABEL: func.func @unary_in_place
// DISABLED-LABEL: func.func @unary_in_place
func.func @unary_in_place() -> tensor<2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_e> : tensor<2xf32>
  // CHECK-NOT: tfl.exp
  // CHECK: return %[[CST]]
  // DISABLED: tfl.exp
  %0 = arith.constant dense_resource<res_e> : tensor<2xf32>
  %1 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// exp([0, 1]) written into the original blob.
// CHECK: res_e: "0x040000000000803F54F82D40"

{-#
  dialect_resources: {
    builtin: {
      res_e: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// A resource that has other users is not mutated; the result gets a new blob.
// CHECK-LABEL: func.func @unary_shared_user
func.func @unary_shared_user() -> (tensor<2xf32>, tensor<2xf32>) {
  // CHECK-DAG: %[[ORIG:.*]] = arith.constant dense_resource<res_s> : tensor<2xf32>
  // CHECK-DAG: %[[NEW:.*]] = arith.constant dense_resource<res_s_exp> : tensor<2xf32>
  // CHECK-NOT: tfl.exp
  // CHECK: return %[[NEW]], %[[ORIG]]
  %0 = arith.constant dense_resource<res_s> : tensor<2xf32>
  %1 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1, %0 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-DAG: res_s: "0x04000000000000000000803F"
// CHECK-DAG: res_s_exp: "0x400000000000803F54F82D40"

{-#
  dialect_resources: {
    builtin: {
      res_s: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// A resource referenced by two constant ops is not mutated even though each
// constant op only has a single user.
// CHECK-LABEL: func.func @unary_resource_referenced_twice
func.func @unary_resource_referenced_twice() -> (tensor<2xf32>, tensor<2xf32>) {
  // CHECK-DAG: %[[ORIG:.*]] = arith.constant dense_resource<res_d> : tensor<2xf32>
  // CHECK-DAG: %[[NEW:.*]] = arith.constant dense_resource<res_d_exp> : tensor<2xf32>
  // CHECK-NOT: tfl.exp
  // CHECK: return %[[NEW]], %[[ORIG]]
  %0 = arith.constant dense_resource<res_d> : tensor<2xf32>
  %1 = arith.constant dense_resource<res_d> : tensor<2xf32>
  %2 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %2, %1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-DAG: res_d: "0x04000000000000000000803F"
// CHECK-DAG: res_d_exp: "0x400000000000803F54F82D40"

{-#
  dialect_resources: {
    builtin: {
      res_d: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// A resource shared across functions is not mutated.
// CHECK-LABEL: func.func @unary_resource_shared_across_functions
func.func @unary_resource_shared_across_functions() -> tensor<2xf32> {
  // CHECK: %[[NEW:.*]] = arith.constant dense_resource<res_f_exp> : tensor<2xf32>
  // CHECK: return %[[NEW]]
  %0 = arith.constant dense_resource<res_f> : tensor<2xf32>
  %1 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @other_function
func.func @other_function() -> tensor<2xf32> {
  // CHECK: %[[ORIG:.*]] = arith.constant dense_resource<res_f> : tensor<2xf32>
  // CHECK: return %[[ORIG]]
  %0 = arith.constant dense_resource<res_f> : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-DAG: res_f: "0x04000000000000000000803F"
// CHECK-DAG: res_f_exp: "0x400000000000803F54F82D40"

{-#
  dialect_resources: {
    builtin: {
      res_f: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// Binary ops broadcast dense and resource operands into a new blob.
// CHECK-LABEL: func.func @binary_broadcast
func.func @binary_broadcast() -> tensor<2x2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_m_mul> : tensor<2x2xf32>
  // CHECK-NOT: tfl.mul
  // CHECK: return %[[CST]]
  %0 = arith.constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %1 = arith.constant dense_resource<res_m> : tensor<1x2xf32>
  %2 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<2x1xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// [[1], [2]] * [[3, 4]] = [[3, 4], [6, 8]].
// CHECK: res_m_mul: "0x4000000000004040000080400000C04000000041"

{-#
  dialect_resources: {
    builtin: {
      res_m: "0x040000000000404000008040"
    }
  }
#-}

// -----

// Rank-3 broadcasting where neither operand is laid out like the output.
// CHECK-LABEL: func.func @binary_broadcast_rank3
func.func @binary_broadcast_rank3() -> tensor<2x2x3xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_b3_add> : tensor<2x2x3xf32>
  // CHECK-NOT: tfl.add
  // CHECK: return %[[CST]]
  %0 = arith.constant dense_resource<res_b3> : tensor<2x1x3xf32>
  %1 = arith.constant dense<[[[10.0], [20.0]]]> : tensor<1x2x1xf32>
  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<2x1x3xf32>, tensor<1x2x1xf32>) -> tensor<2x2x3xf32>
  return %2 : tensor<2x2x3xf32>
}

// [[[10, 11, 12], [20, 21, 22]], [[13, 14, 15], [23, 24, 25]]].
// CHECK: res_b3_add: "0x400000000000204100003041000040410000A0410000A8410000B0410000504100006041000070410000B8410000C0410000C841"

{-#
  dialect_resources: {
    builtin: {
      res_b3: "0x04000000000000000000803F0000004000004040000080400000A040"
    }
  }
#-}

// -----

// f16 arithmetic with a fused activation, computed in place.
// CHECK-LABEL: func.func @binary_f16_relu
func.func @binary_f16_relu() -> tensor<2xf16> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_h> : tensor<2xf16>
  // CHECK-NOT: tfl.add
  // CHECK: return %[[CST]]
  %0 = arith.constant dense_resource<res_h> : tensor<2xf16>
  %1 = arith.constant dense<0.5> : tensor<f16>
  %2 = "tfl.add"(%0, %1) {fused_activation_function = "RELU"} : (tensor<2xf16>, tensor<f16>) -> tensor<2xf16>
  return %2 : tensor<2xf16>
}

// relu([-1, 2] + 0.5) = [0, 2.5].
// CHECK: res_h: "0x0200000000000041"

{-#
  dialect_resources: {
    builtin: {
      res_h: "0x0200000000BC0040"
    }
  }
#-}

// -----

// Select on 2-byte elements reuses the exclusively owned `y` buffer.
// CHECK-LABEL: func.func @select_i16
func.func @select_i16() -> tensor<2xi16> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_y> : tensor<2xi16>
  // CHECK-NOT: tfl.select
  // CHECK: return %[[CST]]
  %cond = arith.constant dense<[false, true]> : tensor<2xi1>
  %x = arith.constant dense<[7, 8]> : tensor<2xi16>
  %y = arith.constant dense_resource<res_y> : tensor<2xi16>
  %0 = "tfl.select"(%cond, %x, %y) : (tensor<2xi1>, tensor<2xi16>, tensor<2xi16>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// [-1, 8].
// CHECK: res_y: "0x02000000FFFF0800"

{-#
  dialect_resources: {
    builtin: {
      res_y: "0x02000000FFFF0200"
    }
  }
#-}

// -----

// Comparisons produce a boolean resource with one byte per element.
// CHECK-LABEL: func.func @compare
func.func @compare() -> tensor<2xi1> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_g_greater> : tensor<2xi1>
  // CHECK-NOT: tfl.greater
  // CHECK: return %[[CST]]
  %0 = arith.constant dense_resource<res_g> : tensor<2xf32>
  %1 = arith.constant dense<0.0> : tensor<f32>
  %2 = "tfl.greater"(%0, %1) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  return %2 : tensor<2xi1>
}

// [1, -1] > 0 = [true, false].
// CHECK: res_g_greater: "0x400000000100"

{-#
  dialect_resources: {
    builtin: {
      res_g: "0x040000000000803F000080BF"
    }
  }
#-}

// -----

// Select reuses the exclusively owned `x` buffer.
// CHECK-LABEL: func.func @select_v2
func.func @select_v2() -> tensor<2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_x> : tensor<2xf32>
  // CHECK-NOT: tfl.select_v2
  // CHECK: return %[[CST]]
  %cond = arith.constant dense<[true, false]> : tensor<2xi1>
  %x = arith.constant dense_resource<res_x> : tensor<2xf32>
  %y = arith.constant dense<20.0> : tensor<f32>
  %0 = "tfl.select_v2"(%cond, %x, %y) : (tensor<2xi1>, tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK: res_x: "0x040000000000803F0000A041"

{-#
  dialect_resources: {
    builtin: {
      res_x: "0x040000000000803F000080BF"
    }
  }
#-}

// -----

// Concatenation of a resource and a dense constant.
// CHECK-LABEL: func.func @concatenation
func.func @concatenation() -> tensor<4xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_a_concatenation> : tensor<4xf32>
  // CHECK-NOT: tfl.concatenation
  // CHECK: return %[[CST]]
  %0 = arith.constant dense_resource<res_a> : tensor<2xf32>
  %1 = arith.constant dense<[5.0, 6.0]> : tensor<2xf32>
  %2 = "tfl.concatenation"(%0, %1) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK: res_a_concatenation: "0x40000000000000000000803F0000A0400000C040"

{-#
  dialect_resources: {
    builtin: {
      res_a: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// A chain of foldable ops collapses in a single run, reusing the same buffer.
// CHECK-LABEL: func.func @chain_in_place
func.func @chain_in_place() -> tensor<2xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_c> : tensor<2xf32>
  // CHECK-NOT: tfl.
  // CHECK: return %[[CST]]
  %0 = arith.constant dense_resource<res_c> : tensor<2xf32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %1 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "tfl.add"(%1, %one) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %3 = "tfl.log"(%2) : (tensor<2xf32>) -> tensor<2xf32>
  return %3 : tensor<2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_c: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// The blob allocated by an out-of-place fold is exclusively owned, so the rest
// of the chain reuses it.
// CHECK-LABEL: func.func @chain_out_of_place
func.func @chain_out_of_place() -> (tensor<2xf32>, tensor<2xf32>) {
  // CHECK-DAG: %[[ORIG:.*]] = arith.constant dense_resource<res_o> : tensor<2xf32>
  // CHECK-DAG: %[[NEW:.*]] = arith.constant dense_resource<res_o_exp> : tensor<2xf32>
  // CHECK-NOT: tfl.
  // CHECK: return %[[NEW]], %[[ORIG]]
  %0 = arith.constant dense_resource<res_o> : tensor<2xf32>
  %1 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "tfl.neg"(%1) : (tensor<2xf32>) -> tensor<2xf32>
  %3 = "tfl.neg"(%2) : (tensor<2xf32>) -> tensor<2xf32>
  return %3, %0 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-DAG: res_o: "0x04000000000000000000803F"
// CHECK-DAG: res_o_exp: "0x400000000000803F54F82D40"

{-#
  dialect_resources: {
    builtin: {
      res_o: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// Ops without resource operands are left to the regular folders.
// CHECK-LABEL: func.func @dense_only_not_folded
func.func @dense_only_not_folded() -> tensor<2xf32> {
  // CHECK: tfl.exp
  %0 = arith.constant dense<[0.0, 1.0]> : tensor<2xf32>
  %1 = "tfl.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----

// Unsupported fused activations are not folded.
// CHECK-LABEL: func.func @unsupported_activation_not_folded
func.func @unsupported_activation_not_folded() -> tensor<2xf32> {
  // CHECK: tfl.add
  %0 = arith.constant dense_resource<res_t> : tensor<2xf32>
  %1 = arith.constant dense<1.0> : tensor<f32>
  %2 = "tfl.add"(%0, %1) {fused_activation_function = "TANH"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_t: "0x04000000000000000000803F"
    }
  }
#-}

// -----

// Non-constant operands are not folded.
// CHECK-LABEL: func.func @non_constant_not_folded
func.func @non_constant_not_folded(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: tfl.mul
  %0 = arith.constant dense_resource<res_n> : tensor<2xf32>
  %1 = "tfl.mul"(%0, %arg0) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_n: "0x04000000000000000000803F"
    }
  }
#-}
