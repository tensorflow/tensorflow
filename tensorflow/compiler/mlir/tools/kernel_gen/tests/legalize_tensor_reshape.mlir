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
// RUN: kernel-gen-opt --split-input-file --legalize-tensor-reshape %s | FileCheck %s

// CHECK-LABEL: func @tanh
func.func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf_entry} {
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = shape.num_elements %0 : tensor<?xindex> -> index
  %from_elements = tensor.from_elements %1 : tensor<1xindex>
  // CHECK: mhlo.dynamic_reshape
  // CHECK-NOT: tensor.reshape
  %2 = tensor.reshape %arg0(%from_elements) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  %3 = mhlo.tanh %2 : tensor<?xf32>
  %4 = tensor.reshape %3(%0) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @integer_pow_loop_ub
func.func @integer_pow_loop_ub(%arg0: i64, %arg1: i64) -> i64 {
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C64:.*]] = arith.constant 64 : index
  // CHECK: scf.for {{.*}} = {{.*}} to %[[C64]] step {{.*}} iter_args({{.*}}) -> (i64, i64, i64)
  %0:3 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %c1_i64, %arg4 = %arg0, %arg5 = %arg1) -> (i64, i64, i64) {
    %1 = arith.andi %arg5, %c1_i64 : i64
    %2 = arith.cmpi eq, %1, %c1_i64 : i64
    %3 = arith.muli %arg3, %arg4 : i64
    %4 = arith.select %2, %3, %arg3 : i64
    %5 = arith.muli %arg4, %arg4 : i64
    %6 = arith.shrui %arg5, %c1_i64 : i64
    scf.yield %4, %5, %6 : i64, i64, i64
  }
  return %0#0 : i64
}

// -----

// CHECK-LABEL: func @index_iter_args_loop_unchanged
func.func @index_iter_args_loop_unchanged(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C6:.*]] = arith.constant 6 : index
  // CHECK: scf.for {{.*}} = {{.*}} to %[[C6]] step {{.*}} iter_args({{.*}}) -> (index, index, index)
  %0:3 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %arg0, %arg4 = %arg0, %arg5 = %arg1) -> (index, index, index) {
    scf.yield %arg3, %arg4, %arg5 : index, index, index
  }
  return %0#0 : index
}

// -----

// CHECK-LABEL: func @non_pow_integer_loop_unchanged
func.func @non_pow_integer_loop_unchanged(%arg0: i64, %arg1: i64) -> i64 {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C6:.*]] = arith.constant 6 : index
  // CHECK: scf.for {{.*}} = {{.*}} to %[[C6]] step {{.*}} iter_args({{.*}}) -> (i64, i64, i64)
  %0:3 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %arg0, %arg4 = %arg0, %arg5 = %arg1) -> (i64, i64, i64) {
    scf.yield %arg3, %arg4, %arg5 : i64, i64, i64
  }
  return %0#0 : i64
}

// -----

// CHECK-LABEL: func @integer_pow_loop_ub_i16
func.func @integer_pow_loop_ub_i16(%arg0: i16, %arg1: i16) -> i16 {
  %c1_i16 = arith.constant 1 : i16
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: scf.for {{.*}} = {{.*}} to %[[C16]] step {{.*}} iter_args({{.*}}) -> (i16, i16, i16)
  %0:3 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %c1_i16, %arg4 = %arg0, %arg5 = %arg1) -> (i16, i16, i16) {
    %1 = arith.andi %arg5, %c1_i16 : i16
    %2 = arith.cmpi eq, %1, %c1_i16 : i16
    %3 = arith.muli %arg3, %arg4 : i16
    %4 = arith.select %2, %3, %arg3 : i16
    %5 = arith.muli %arg4, %arg4 : i16
    %6 = arith.shrui %arg5, %c1_i16 : i16
    scf.yield %4, %5, %6 : i16, i16, i16
  }
  return %0#0 : i16
}

// -----

// CHECK-LABEL: func @non_matching_operands_loop_unchanged
func.func @non_matching_operands_loop_unchanged(%arg0: i64, %arg1: i64) -> i64 {
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C6:.*]] = arith.constant 6 : index
  // CHECK: scf.for {{.*}} = {{.*}} to %[[C6]] step {{.*}} iter_args({{.*}}) -> (i64, i64, i64)
  %0:3 = scf.for %arg2 = %c0 to %c6 step %c1 iter_args(%arg3 = %c1_i64, %arg4 = %arg0, %arg5 = %arg1) -> (i64, i64, i64) {
    %1 = arith.andi %arg5, %c1_i64 : i64
    %2 = arith.cmpi eq, %1, %c1_i64 : i64
    %3 = arith.muli %arg3, %arg4 : i64
    %4 = arith.select %2, %3, %arg3 : i64
    %5 = arith.muli %arg3, %arg3 : i64
    %6 = arith.shrui %arg5, %c1_i64 : i64
    scf.yield %4, %5, %6 : i64, i64, i64
  }
  return %0#0 : i64
}
