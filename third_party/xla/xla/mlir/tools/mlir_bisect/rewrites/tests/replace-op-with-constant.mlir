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
// RUN: mlir-bisect %s --debug-strategy=ReplaceOpWithConstant | FileCheck %s

func.func @main() -> tensor<2xi32> {
  %a = arith.constant dense<3> : tensor<2xi32>
  %b = arith.constant dense<2> : tensor<2xi32>
  %c = mhlo.add %a, %b : tensor<2xi32>
  %d = mhlo.multiply %b, %c : tensor<2xi32>
  func.return %d : tensor<2xi32>
}

//      CHECK: func.func @main()
// CHECK-NEXT:   arith.constant dense<3>
// CHECK-NEXT:   arith.constant dense<2>
// CHECK-NEXT:   arith.constant dense<5>
// CHECK-NEXT:   %[[ADD:.*]] = mhlo.add
//  CHECK-NOT:   %[[ADD]]
// CHECK-NEXT:   mhlo.multiply
// CHECK-NEXT:   return

//      CHECK: func.func @main()
// CHECK-NEXT:   arith.constant dense<3>
// CHECK-NEXT:   arith.constant dense<2>
// CHECK-NEXT:   mhlo.add
// CHECK-NEXT:   %[[D:.*]] = arith.constant dense<10>
// CHECK-NEXT:   mhlo.multiply
// CHECK-NEXT:   return %[[D]]
