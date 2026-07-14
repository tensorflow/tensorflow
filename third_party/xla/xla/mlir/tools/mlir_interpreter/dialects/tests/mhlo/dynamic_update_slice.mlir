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
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dynamic_update_slice() -> tensor<3x4xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %v = mhlo.constant dense<[[13, 14]]> : tensor<1x2xi32>
  %s0 = mhlo.constant dense<1> : tensor<i32>
  %s1 = mhlo.constant dense<0> : tensor<i32>
  %0 = "mhlo.dynamic_update_slice"(%cst, %v, %s0, %s1)
    : (tensor<3x4xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>)
      -> tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
}

// CHECK-LABEL: @dynamic_update_slice
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 3, 4], [13, 14, 7, 8], [9, 10, 11, 12]]

func.func @clamp_starts() -> tensor<3x4xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %v = mhlo.constant dense<[[13, 14]]> : tensor<1x2xi32>
  %s0 = mhlo.constant dense<-10> : tensor<i32>
  %s1 = mhlo.constant dense<10> : tensor<i32>
  %0 = "mhlo.dynamic_update_slice"(%cst, %v, %s0, %s1) {
    slice_sizes = dense<[2, 3]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>)
      -> tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
}

// CHECK-LABEL: @clamp_starts
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 13, 14], [5, 6, 7, 8], [9, 10, 11, 12]]
