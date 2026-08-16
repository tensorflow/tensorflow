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
// RUN: tf-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// Test that a constant stays inside an island after canonicalization

// CHECK-LABEL: func @constant_in_island
func.func @constant_in_island(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.island
// CHECK: tf.Const{{.*}}2.0
    %1:2 = tf_executor.island {
      %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
      tf_executor.yield %0 : tensor<f32>
    }
// Uses two islands for no other reason than preventing canonicalization from
// eliminating the graph entirely.
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.opB"(%1#0) : (tensor<f32>) -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }
    tf_executor.fetch %2#0 : tensor<f32>
  }
  func.return %0 : tensor<f32>
}
