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

func.func @compressstore() -> memref<3x4xi32> {
  %alloc = memref.alloc() : memref<3x4xi32>
  %c = arith.constant dense<[1,2,3]> : vector<3xi32>
  %m = arith.constant dense<[true,false,true]> : vector<3xi1>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  vector.compressstore %alloc[%c1, %c2], %m, %c
    : memref<3x4xi32>, vector<3xi1>, vector<3xi32>
  return %alloc : memref<3x4xi32>
}

// CHECK-LABEL: @compressstore
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 0, 1, 3], [0, 0, 0, 0]]
