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

func.func @case() -> tensor<i32> {
  %c1 = mhlo.constant dense<1> : tensor<i32>
  %c2 = mhlo.constant dense<2> : tensor<i32>
  %c3 = mhlo.constant dense<3> : tensor<i32>
  %ret = "mhlo.case"(%c1) ({
    "mhlo.return"(%c2) : (tensor<i32>) -> ()
  }, {
    "mhlo.return"(%c3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  func.return %ret : tensor<i32>
}

// CHECK-LABEL: @case
// CHECK-NEXT: Results
// CHECK-NEXT: <i32>: 3
