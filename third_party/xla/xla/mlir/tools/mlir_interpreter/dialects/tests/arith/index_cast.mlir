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

func.func @i32() -> (index) {
  %c1 = arith.constant 42 : i32
  %index = arith.index_cast %c1 : i32 to index
  return %index : index
}

// CHECK-LABEL: @i32
// CHECK{LITERAL}: 42

func.func @i64() -> (index) {
  %c1 = arith.constant 43 : i64
  %index = arith.index_cast %c1 : i64 to index
  return %index : index
}

// CHECK-LABEL: @i64
// CHECK{LITERAL}: 43

func.func @narrowing() -> (i32) {
  %c1 = arith.constant 0x100000001 : index
  %i32 = arith.index_cast %c1 : index to i32
  return %i32 : i32
}

// CHECK-LABEL: @narrowing
// CHECK{LITERAL}: i32: 1
