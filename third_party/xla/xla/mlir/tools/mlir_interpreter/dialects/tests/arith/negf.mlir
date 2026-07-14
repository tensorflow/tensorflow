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

func.func @negf32() -> f32 {
  %c = arith.constant -1.5 : f32
  %ret = arith.negf %c : f32
  return %ret : f32
}

// CHECK-LABEL: @negf32
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 1.500000e+00

func.func @negf64() -> f64 {
  %c = arith.constant 3.5 : f64
  %ret = arith.negf %c : f64
  return %ret : f64
}

// CHECK-LABEL: @negf64
// CHECK-NEXT: Results
// CHECK-NEXT: f64: -3.500000e+00
