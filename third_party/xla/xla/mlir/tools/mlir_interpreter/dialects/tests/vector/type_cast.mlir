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

func.func @type_cast() -> memref<vector<2x2xi32>> {
  %alloc = arith.constant dense<[[1, 2], [3, 4]]> : memref<2x2xi32>
  %cast = vector.type_cast %alloc: memref<2x2xi32> to memref<vector<2x2xi32>>
  return %cast : memref<vector<2x2xi32>>
}

// CHECK-LABEL: @type_cast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<vector<2x2xi32>>: [[1, 2], [3, 4]]
