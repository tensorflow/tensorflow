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

func.func @concat_1d()  -> tensor<3xi32> {
  %a = mhlo.constant dense<[1]> : tensor<1xi32>
  %b = mhlo.constant dense<[2, 3]> : tensor<2xi32>
  %0 = "mhlo.concatenate"(%a, %b) { dimension = 0 : i64 }
     : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @concat_1d
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 2, 3]

func.func @concat_dim0() -> tensor<4x2xi32> {
  %a = mhlo.constant dense<1> : tensor<2x2xi32>
  %b = mhlo.constant dense<2> : tensor<2x2xi32>
  %0 = "mhlo.concatenate"(%a, %b) { dimension = 0 : i64 }
     : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  func.return %0 : tensor<4x2xi32>
}

// CHECK-LABEL: @concat_dim0
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1], [1, 1], [2, 2], [2, 2]]

func.func @concat_dim1() -> tensor<2x4xi32> {
  %a = mhlo.constant dense<1> : tensor<2x2xi32>
  %b = mhlo.constant dense<2> : tensor<2x2xi32>
  %0 = "mhlo.concatenate"(%a, %b) { dimension = 1 : i64 }
     : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// CHECK-LABEL: @concat_dim1
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1, 2, 2], [1, 1, 2, 2]]
