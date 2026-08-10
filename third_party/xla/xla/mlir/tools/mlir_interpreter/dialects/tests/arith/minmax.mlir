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

func.func @i32() -> (i32, i32) {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32
  %r1 = arith.minsi %c-1, %c1 : i32
  %r2 = arith.maxsi %c-1, %c1 : i32
  return %r1, %r2 : i32, i32
}

// CHECK-LABEL: @i32
// CHECK{LITERAL}: -1
// CHECK-NEXT{LITERAL}: 1

func.func @i64() -> (i64, i64) {
  %c-1 = arith.constant -1 : i64
  %c1 = arith.constant 1000000000000 : i64
  %r1 = arith.minsi %c-1, %c1 : i64
  %r2 = arith.maxsi %c-1, %c1 : i64
  return %r1, %r2 : i64, i64
}

// CHECK-LABEL: @i64
// CHECK{LITERAL}: -1
// CHECK-NEXT{LITERAL}: 1000000000000
