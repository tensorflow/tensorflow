// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// RUN: mlir-hlo-opt %s -inline | FileCheck %s

// Test case: Basic test of inlining into mhlo.while.

// CHECK-LABEL: func @caller
// CHECK:   mhlo.while
// CHECK:     mhlo.exponential

// CHECK-LABEL: func @callee

func.func @caller(%arg0: tensor<f32>, %pred: tensor<i1>) -> tensor<f32> {
  %0 = "mhlo.while"(%arg0) ({
  ^entry(%unused: tensor<f32>):
    "mhlo.return"(%pred) : (tensor<i1>) -> ()
  }, {
  ^entry(%0: tensor<f32>):
    %1 = func.call @callee(%0) : (tensor<f32>) -> (tensor<f32>)
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  } ) : (tensor<f32>) -> (tensor<f32>)
  func.return %0 : tensor<f32>
}


func.func @callee(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.exponential %arg0 : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
