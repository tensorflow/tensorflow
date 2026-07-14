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

func.func @alloc() -> memref<2x3xi32> {
  %ret = memref.alloc() : memref<2x3xi32>
  return %ret : memref<2x3xi32>
}

// CHECK-LABEL: @alloc
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <2x3xi32>: [[0, 0, 0], [0, 0, 0]]

func.func @alloc_unit() -> memref<i32> {
  %ret = memref.alloc() : memref<i32>
  return %ret : memref<i32>
}

// CHECK-LABEL: @alloc_unit
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <i32>: 0

func.func @alloc_unit_vector() -> memref<vector<i32>> {
  %ret = memref.alloc() : memref<vector<i32>>
  return %ret : memref<vector<i32>>
}

// CHECK-LABEL: @alloc_unit_vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <vector<i32>>: 0

func.func @alloc_vector() -> memref<2xvector<3xi32>> {
  %ret = memref.alloc() : memref<2xvector<3xi32>>
  return %ret : memref<2xvector<3xi32>>
}

// CHECK-LABEL: @alloc_vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <2xvector<3xi32>>: [[0, 0, 0], [0, 0, 0]]

func.func @alloc_dynamic() -> memref<?x3xi32> {
  %c2 = arith.constant 2 : index
  %ret = memref.alloc(%c2) : memref<?x3xi32>
  return %ret : memref<?x3xi32>
}

// CHECK-LABEL: @alloc_dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <2x3xi32>: [[0, 0, 0], [0, 0, 0]]

func.func @dealloc() -> memref<i32> {
  %a = memref.alloc() : memref<i32>
  memref.dealloc %a : memref<i32>
  return %a : memref<i32>
}

// CHECK-LABEL: @dealloc
// CHECK-NEXT: Results
// CHECK-NEXT: TensorOrMemref<i32>: <<deallocated>>
