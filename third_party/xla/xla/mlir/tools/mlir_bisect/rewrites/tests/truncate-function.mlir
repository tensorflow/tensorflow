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
// RUN: mlir-bisect %s --debug-strategy=TruncateFunction | FileCheck %s

// Function to prevent constant folding below.
func.func private @cst() -> tensor<2xi32> {
  %cst = arith.constant dense<2> : tensor<2xi32>
  return %cst : tensor<2xi32>
}

func.func @main() -> tensor<2xi32> {
  %a = arith.constant dense<1> : tensor<2xi32>
  %b = func.call @cst() : () -> tensor<2xi32>
  %c = mhlo.add %a, %b : tensor<2xi32>
  %d = mhlo.multiply %b, %c : tensor<2xi32>
  func.return %d : tensor<2xi32>
}

//     CHECK: func @main()
//     CHECK:   %[[A:.*]] = arith.constant dense<1>
//     CHECK:   return %[[A]]

//     CHECK: func @main()
//     CHECK:   %[[B:.*]] = call @cst()
//     CHECK:   return %[[B]]

//     CHECK: func @main()
//     CHECK:   %[[A:.*]] = arith.constant dense<1>
//     CHECK:   %[[B:.*]] = call @cst()
//     CHECK:   %[[ADD:.*]] = mhlo.add
// CHECK-DAG:   %[[A]]
// CHECK-DAG:   %[[B]]
//     CHECK:   return %[[ADD]]
