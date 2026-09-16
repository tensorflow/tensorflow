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
// RUN: litert-opt %s --tf-guarantee-all-funcs-one-use --tf-shape-inference | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: func @while_main
// CHECK: %0 = "tf.While"(%arg0)
// CHECK-SAME: body = @while_body
// CHECK-SAME: cond = @while_cond
// CHECK: "tf.While"(%arg1)
// CHECK-SAME: body = @while_body_0
// CHECK-SAME: cond = @while_cond_1

// CHECK: func @while_body(%arg0: tensor<256x256xi32>)
// CHECK: func @while_cond(%arg0: tensor<256x256xi32>)
// CHECK: func private @while_body_0(%arg0: tensor<128xi32>)
// CHECK: func private @while_cond_1(%arg0: tensor<128xi32>)
func.func @while_main(%arg0: tensor<256x256xi32>, %arg1: tensor<128xi32>) -> (tensor<256x256xi32>, tensor<128xi32>) {
  %0 = "tf.While"(%arg0) {body = @while_body, cond = @while_cond, device = "", is_stateless = true} : (tensor<256x256xi32>) -> (tensor<256x256xi32>)
  %1 = "tf.While"(%arg1) {body = @while_body, cond = @while_cond, device = "", is_stateless = true} : (tensor<128xi32>) -> (tensor<128xi32>)
  func.return %0, %1: tensor<256x256xi32>, tensor<128xi32>
}

func.func @while_body(%arg0: tensor<*xi32>) -> (tensor<*xi32>) {
  %0 = "tf.Rank"(%arg0) : (tensor<*xi32>) -> tensor<i32>
  %1 = "tf.Add"(%0, %arg0): (tensor<i32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %1: tensor<*xi32>
}

func.func @while_cond(%arg0: tensor<*xi32>) -> tensor<*xi1> {
  %cst = arith.constant dense<10> : tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) {T = i32, device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}
}
