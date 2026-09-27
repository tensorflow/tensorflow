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

func.func @i16() -> i16 {
  %c-1 = arith.constant -1.4 : f32
  %r = arith.fptosi %c-1 : f32 to i16
  return %r : i16
}

// CHECK-LABEL: @i16
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: -1

func.func @vector() -> vector<2xi32> {
  %c = arith.constant dense<[-1.1, -0.9]> : vector<2xf32>
  %r = arith.fptosi %c : vector<2xf32> to vector<2xi32>
  return %r : vector<2xi32>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2xi32>: [-1, 0]
