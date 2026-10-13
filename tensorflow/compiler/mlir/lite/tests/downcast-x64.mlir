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
// RUN: litert-opt %s --tfl-downcast-x64 --canonicalize -verify-diagnostics | FileCheck %s

// CHECK-LABEL: testFuncSignature
// CHECK: (%arg0: tensor<4xi32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32>
func.func @testFuncSignature(%arg0: tensor<4xi64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: return %arg1 : tensor<2x2xf32>
  func.return %arg1 : tensor<2x2xf64>
}

// CHECK-LABEL: testConstantDowncast
func.func @testConstantDowncast() -> tensor<2xf64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK: return %[[CST]]
  %0 = arith.constant dense<[1.0, 2.0]> : tensor<2xf64>
  func.return %0 : tensor<2xf64>
}

// CHECK-LABEL: testI64ConstantDowncast
func.func @testI64ConstantDowncast() -> tensor<i64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<42> : tensor<i32>
  // CHECK: return %[[CST]]
  %0 = arith.constant dense<42> : tensor<i64>
  func.return %0 : tensor<i64>
}

// INT64_MAX / INT64_MIN sentinels saturate to INT32_MAX / INT32_MIN instead of
// wrapping to -1 / 0.
// CHECK-LABEL: testI64ConstantSaturates
func.func @testI64ConstantSaturates() -> tensor<4xi64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[2147483647, -2147483648, 2147483647, -2147483648]> : tensor<4xi32>
  // CHECK: return %[[CST]]
  // expected-warning @+1 {{saturating them to [INT32_MIN, INT32_MAX]}}
  %0 = arith.constant dense<[9223372036854775807, -9223372036854775808, 2147483648, -2147483649]> : tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// CHECK-LABEL: testI64SplatConstantSaturates
func.func @testI64SplatConstantSaturates() -> tensor<2x3xi64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<2147483647> : tensor<2x3xi32>
  // CHECK: return %[[CST]]
  // expected-warning @+1 {{saturating them to [INT32_MIN, INT32_MAX]}}
  %0 = arith.constant dense<9223372036854775807> : tensor<2x3xi64>
  func.return %0 : tensor<2x3xi64>
}

// CHECK-LABEL: testTflConstSaturates
func.func @testTflConstSaturates() -> tensor<2xi64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[-2147483648, 7]> : tensor<2xi32>
  // CHECK: return %[[CST]]
  // expected-warning @+1 {{saturating them to [INT32_MIN, INT32_MAX]}}
  %0 = "tfl.pseudo_const"() {value = dense<[-9223372036854775808, 7]> : tensor<2xi64>} : () -> tensor<2xi64>
  func.return %0 : tensor<2xi64>
}

// Unsigned constants clamp to [0, UINT32_MAX] with unsigned comparisons: a
// value with the top bit set saturates to UINT32_MAX rather than being read as
// negative and clamped to the bottom of the range. (arith.constant requires
// signless types, so these use tfl.pseudo_const.)
// CHECK-LABEL: testUI64ConstantSaturates
func.func @testUI64ConstantSaturates() -> tensor<3xui64> {
  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[4294967295, 4294967295, 7]> : tensor<3xui32>}> : () -> tensor<3xui32>
  // CHECK: return %[[CST]]
  // expected-warning @+1 {{saturating them to [0, UINT32_MAX]}}
  %0 = "tfl.pseudo_const"() {value = dense<[18446744073709551615, 4294967296, 7]> : tensor<3xui64>} : () -> tensor<3xui64>
  func.return %0 : tensor<3xui64>
}

// Unsigned values that fit in 32 bits but exceed INT32_MAX are kept.
// CHECK-LABEL: testUI64ConstantInRangeNoWarning
func.func @testUI64ConstantInRangeNoWarning() -> tensor<2xui64> {
  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[4294967295, 2147483648]> : tensor<2xui32>}> : () -> tensor<2xui32>
  // CHECK: return %[[CST]]
  %0 = "tfl.pseudo_const"() {value = dense<[4294967295, 2147483648]> : tensor<2xui64>} : () -> tensor<2xui64>
  func.return %0 : tensor<2xui64>
}

// ui64 function arguments narrow to ui32, preserving signedness.
// CHECK-LABEL: testUI64FuncSignature
// CHECK: (%arg0: tensor<4xui32>) -> tensor<4xui32>
func.func @testUI64FuncSignature(%arg0: tensor<4xui64>) -> tensor<4xui64> {
  // CHECK: return %arg0 : tensor<4xui32>
  func.return %arg0 : tensor<4xui64>
}

// In-range values pass through unchanged and emit no warning.
// CHECK-LABEL: testI64ConstantInRangeNoWarning
func.func @testI64ConstantInRangeNoWarning() -> tensor<2xi64> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[2147483647, -2147483648]> : tensor<2xi32>
  // CHECK: return %[[CST]]
  %0 = arith.constant dense<[2147483647, -2147483648]> : tensor<2xi64>
  func.return %0 : tensor<2xi64>
}

// CHECK-LABEL: testGeneralOpForceConvert
func.func @testGeneralOpForceConvert(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  // CHECK: %[[ADD:.*]] = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2xf32>
  // CHECK: return %[[ADD]]
  %0 = "tfl.add"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  func.return %0 : tensor<2xf64>
}

// Block arguments of nested regions (here a region-based tfl.while, which is
// only outlined into functions after this pass runs) are narrowed along with
// the operands feeding them.
// CHECK-LABEL: testWhileRegionBlockArgs
// CHECK-SAME: (%arg0: tensor<i32>) -> tensor<i32>
func.func @testWhileRegionBlockArgs(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: "tfl.while"(%arg0) ({
  // CHECK: ^bb0(%[[CARG:.*]]: tensor<i32>):
  // CHECK: tfl.less(%[[CARG]], %{{.*}}) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: ^bb0(%[[BARG:.*]]: tensor<i32>):
  // CHECK: tfl.add %[[BARG]], %{{.*}} {fused_activation_function = "NONE"} : tensor<i32>
  // CHECK: }) : (tensor<i32>) -> tensor<i32>
  %0 = "tfl.while"(%arg0) ({
    ^bb0(%carg: tensor<i64>):
      %limit = arith.constant dense<10> : tensor<i64>
      %cond = "tfl.less"(%carg, %limit) : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "tfl.yield"(%cond) : (tensor<i1>) -> ()
  }, {
    ^bb0(%barg: tensor<i64>):
      %one = arith.constant dense<1> : tensor<i64>
      %next = "tfl.add"(%barg, %one) {fused_activation_function = "NONE"} : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "tfl.yield"(%next) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// Resource-backed constants (what lightweight_conversion produces for large
// tensors) are narrowed into a new blob with the same saturation rules.
// res_i64 = [INT64_MAX, INT64_MIN, 5].
// CHECK-LABEL: testI64ResourceConstantSaturates
func.func @testI64ResourceConstantSaturates() -> tensor<3xi64> {
  // CHECK: %[[CST:.*]] = arith.constant dense_resource<res_i64_x32> : tensor<3xi32>
  // CHECK: return %[[CST]]
  // expected-warning @+1 {{saturating them to [INT32_MIN, INT32_MAX]}}
  %0 = arith.constant dense_resource<res_i64> : tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// res_ui64 = [UINT64_MAX, 7].
// CHECK-LABEL: testUI64ResourceConstantSaturates
func.func @testUI64ResourceConstantSaturates() -> tensor<2xui64> {
  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense_resource<res_ui64_x32> : tensor<2xui32>}> : () -> tensor<2xui32>
  // CHECK: return %[[CST]]
  // expected-warning @+1 {{saturating them to [0, UINT32_MAX]}}
  %0 = "tfl.pseudo_const"() {value = dense_resource<res_ui64> : tensor<2xui64>} : () -> tensor<2xui64>
  func.return %0 : tensor<2xui64>
}

// The narrowed blobs: 64-byte alignment prefix, then
// [INT32_MAX, INT32_MIN, 5] and [UINT32_MAX, 7] as little-endian 32-bit.
// CHECK-DAG: res_i64_x32: "0x40000000FFFFFF7F0000008005000000"
// CHECK-DAG: res_ui64_x32: "0x40000000FFFFFFFF07000000"

{-#
  dialect_resources: {
    builtin: {
      res_i64: "0x08000000FFFFFFFFFFFFFF7F00000000000000800500000000000000",
      res_ui64: "0x08000000FFFFFFFFFFFFFFFF0700000000000000"
    }
  }
#-}