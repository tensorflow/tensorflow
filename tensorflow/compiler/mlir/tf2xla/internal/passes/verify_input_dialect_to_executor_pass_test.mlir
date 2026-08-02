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
// RUN: tf-opt -verify-input-dialect-to-executor-pass  -split-input-file -verify-diagnostics %s | FileCheck %s
// Tests the VerifyClusteringPass Pass, ensures that an error is thrown when validation fails.

// -----

// CHECK-LABEL: func @testNoClusterFuncOpPasses
func.func @testNoClusterFuncOpPasses(%arg0: tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string> {
  %0 = "tf.Identity"(%arg0) : (tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}

// -----

func.func @_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

func.func @testClusterFuncOpFails(%arg0: tensor<i32>) -> tensor<i32> {
   // expected-error@below {{failed TF functional to executor validation, op tf_device.cluster_func is not allowed}}
  %cluster = "tf_device.cluster_func"(%arg0) {func = @_func} : (tensor<i32>) -> tensor<i32>
 func.return %cluster : tensor<i32>
}

// -----

// CHECK-LABEL: func @testTFDialect
func.func @testTFDialect(%arg0: tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string> {
  %0 = "tf.Identity"(%arg0) : (tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}

// -----

func.func @testNotTfDialect(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
 // expected-error@below {{op is in dialect chlo which is not an accepted dialect}}
  %0 = "chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = array<i64: 3>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}
