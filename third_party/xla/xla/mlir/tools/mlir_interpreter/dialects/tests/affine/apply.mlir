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

func.func @ceildiv() -> index {
  %c25 = arith.constant 25 : index
  %ret = affine.apply affine_map<(d0)[] -> ((d0 ceildiv 8) * 8)>(%c25)[]
  return %ret : index
}

// CHECK-LABEL: @ceildiv
// CHECK: Results
// CHECK-NEXT: 32

func.func @floordiv() -> index {
  %c25 = arith.constant 25 : index
  %ret = affine.apply affine_map<(d0)[] -> ((d0 floordiv 8) * 8)>(%c25)[]
  return %ret : index
}

// CHECK-LABEL: @floordiv
// CHECK: Results
// CHECK-NEXT: 24

func.func @add() -> index {
  %c100 = arith.constant 100 : index
  %c42 = arith.constant 42 : index
  %ret = affine.apply affine_map<(d0, d1)[] -> (d0 + d1)>(%c100, %c42)[]
  return %ret : index
}

// CHECK-LABEL: @add
// CHECK: Results
// CHECK-NEXT: 142

func.func @mod() -> index {
  %c99 = arith.constant 99 : index
  %ret = affine.apply affine_map<(d0)[] -> (d0 mod 10)>(%c99)[]
  return %ret : index
}

// CHECK-LABEL: @mod
// CHECK: Results
// CHECK-NEXT: 9

func.func @mul() -> index {
  %c100 = arith.constant 100 : index
  %ret = affine.apply affine_map<(d0)[] -> (d0 * 42)>(%c100)[]
  return %ret : index
}

// CHECK-LABEL: @mul
// CHECK: Results
// CHECK-NEXT: 4200

func.func @symbol() -> index {
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %ret = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c1)[%c20]
  return %ret : index
}

// CHECK-LABEL: @symbol
// CHECK: Results
// CHECK-NEXT: 21
