// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s | tf-opt | FileCheck %s

// Tests printer for tf_executor.island "wraps" short form.

// CHECK-LABEL: func @island_wrap_print
func.func @island_wrap_print(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island wraps "tf.IdentityN"
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %1#0, %1#1 : tensor<i32>, tensor<f32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  func.return
}

// CHECK-LABEL: func @island_no_wrap_print_mismatched_results
func.func @island_no_wrap_print_mismatched_results(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %1#1, %1#0 : tensor<f32>, tensor<i32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  func.return
}

// CHECK-LABEL: func @island_no_wrap_print_mismatched_op_location
func.func @island_no_wrap_print_mismatched_op_location(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc(unknown)
      tf_executor.yield %1#0, %1#1 : tensor<i32>, tensor<f32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  func.return
}

// CHECK-LABEL: func @island_no_wrap_print_mismatched_yield_location
func.func @island_no_wrap_print_mismatched_yield_location(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %1#0, %1#1 : tensor<i32>, tensor<f32> loc(unknown)
    } loc("identity@some_function")
    tf_executor.fetch
  }
  func.return
}

// CHECK-LABEL: func @island_no_wrap_print_multiple_ops
func.func @island_no_wrap_print_multiple_ops(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      %2:2 = "tf.IdentityN"(%1#0, %1#1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %2#0, %2#1 : tensor<i32>, tensor<f32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  func.return
}
