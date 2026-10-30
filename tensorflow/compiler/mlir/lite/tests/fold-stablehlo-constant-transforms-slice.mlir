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

// RUN: litert-opt %s -tfl-fold-stablehlo-constant-transforms -split-input-file | FileCheck %s

// Unit-stride slice of a resource constant folds into a new resource constant.
// CHECK-LABEL: func.func @slice_resource
func.func @slice_resource() -> tensor<1x3xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense_resource<stablehlo_slice_fold_result_{{[0-9]+}}> : tensor<1x3xf32>
  // CHECK-NOT: stablehlo.slice
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense_resource<res_2x3> : tensor<2x3xf32>
  %1 = "stablehlo.slice"(%0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 3>, strides = array<i64: 1, 1>} : (tensor<2x3xf32>) -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// Values [3, 4, 5].
// CHECK: stablehlo_slice_fold_result_{{[0-9]+}}: "0x4000000000004040000080400000A040"

{-#
  dialect_resources: {
    builtin: {
      res_2x3: "0x40000000000000000000803F0000004000004040000080400000A040"
    }
  }
#-}

// -----

// Strided slice of a resource constant folds into a new resource constant.
// CHECK-LABEL: func.func @strided_slice_resource
func.func @strided_slice_resource() -> tensor<2x2xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense_resource<stablehlo_slice_fold_result_{{[0-9]+}}> : tensor<2x2xf32>
  // CHECK-NOT: stablehlo.slice
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense_resource<res_2x3> : tensor<2x3xf32>
  %1 = "stablehlo.slice"(%0) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 3>, strides = array<i64: 1, 2>} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// Values [[0, 2], [3, 5]].
// CHECK: stablehlo_slice_fold_result_{{[0-9]+}}: "0x400000000000000000000040000040400000A040"

{-#
  dialect_resources: {
    builtin: {
      res_2x3: "0x40000000000000000000803F0000004000004040000080400000A040"
    }
  }
#-}

// -----

// A resource constant with multiple slice users folds every slice, and the
// following reshapes fold as well by reusing the slice result blobs.
// CHECK-LABEL: func.func @slice_resource_multiple_users
func.func @slice_resource_multiple_users() -> (tensor<3xf32>, tensor<3xf32>) {
  // CHECK-DAG: %[[CST0:.*]] = stablehlo.constant dense_resource<stablehlo_slice_fold_result_{{[0-9]+}}> : tensor<3xf32>
  // CHECK-DAG: %[[CST1:.*]] = stablehlo.constant dense_resource<stablehlo_slice_fold_result_{{[0-9]+}}> : tensor<3xf32>
  // CHECK-NOT: stablehlo.slice
  // CHECK-NOT: stablehlo.reshape
  %0 = stablehlo.constant dense_resource<res_2x3> : tensor<2x3xf32>
  %1 = "stablehlo.slice"(%0) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 3>, strides = array<i64: 1, 1>} : (tensor<2x3xf32>) -> tensor<1x3xf32>
  %2 = "stablehlo.slice"(%0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 3>, strides = array<i64: 1, 1>} : (tensor<2x3xf32>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %1 : (tensor<1x3xf32>) -> tensor<3xf32>
  %4 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3xf32>
  return %3, %4 : tensor<3xf32>, tensor<3xf32>
}

{-#
  dialect_resources: {
    builtin: {
      res_2x3: "0x40000000000000000000803F0000004000004040000080400000A040"
    }
  }
#-}

// -----

// Slices of non-constant operands are left untouched.
// CHECK-LABEL: func.func @slice_non_constant
func.func @slice_non_constant(%arg0: tensor<2x3xf32>) -> tensor<1x3xf32> {
  // CHECK: stablehlo.slice
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 3>, strides = array<i64: 1, 1>} : (tensor<2x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// -----

// Sub-byte resource constants are not folded.
// CHECK-LABEL: func.func @slice_resource_sub_byte
func.func @slice_resource_sub_byte() -> tensor<1x4xi4> {
  // CHECK: stablehlo.slice
  %0 = stablehlo.constant dense_resource<res_i4> : tensor<2x4xi4>
  %1 = "stablehlo.slice"(%0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<2x4xi4>) -> tensor<1x4xi4>
  return %1 : tensor<1x4xi4>
}

{-#
  dialect_resources: {
    builtin: {
      res_i4: "0x010000000102030405060708"
    }
  }
#-}
