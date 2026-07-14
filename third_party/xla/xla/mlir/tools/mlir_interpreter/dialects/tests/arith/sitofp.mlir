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

func.func @i16() -> f32 {
  %c-1 = arith.constant -1 : i16
  %r = arith.sitofp %c-1 : i16 to f32
  return %r : f32
}

// CHECK-LABEL: @i16
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: -1.000000e+00

func.func @i1() -> f64 {
  %true = arith.constant true
  %r = arith.sitofp %true : i1 to f64
  return %r : f64
}

// CHECK-LABEL: @i1
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: 1.000000e+00

func.func @vector() -> vector<1xf32> {
  %c-1 = arith.constant dense<-1> : vector<1xi8>
  %r = arith.sitofp %c-1 : vector<1xi8> to vector<1xf32>
  return %r : vector<1xf32>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<1xf32>: [-1.000000e+00]
