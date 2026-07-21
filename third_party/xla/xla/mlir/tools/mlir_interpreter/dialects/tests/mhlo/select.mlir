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

func.func @reshape() -> tensor<2xi32> {
  %cst = mhlo.constant dense<[true, false]> : tensor<2xi1>
  %a = mhlo.constant dense<[1, 2]> : tensor<2xi32>
  %b = mhlo.constant dense<[3, 4]> : tensor<2xi32>
  %ret = "mhlo.select"(%cst, %a, %b) :
    (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %ret : tensor<2xi32>
}

// CHECK-LABEL: @reshape
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 4]
