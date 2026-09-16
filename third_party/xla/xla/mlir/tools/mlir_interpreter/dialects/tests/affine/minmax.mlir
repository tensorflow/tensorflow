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

#map = affine_map<(d0)[] -> (1000, d0 + 512, d0*100)>

func.func @min() -> (index, index, index) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c500 = arith.constant 500 : index

  %0 = affine.min #map (%c1)[]
  %1 = affine.min #map (%c8)[]
  %2 = affine.min #map (%c500)[]

  return %0, %1, %2 : index, index, index
}

// CHECK-LABEL: @min
// CHECK-NEXT: Results
// CHECK-NEXT: 100
// CHECK-NEXT: 520
// CHECK-NEXT: 1000

func.func @max() -> (index, index) {
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index

  %0 = affine.max #map (%c1)[]
  %1 = affine.max #map (%c11)[]

  return %0, %1 : index, index
}

// CHECK-LABEL: @max
// CHECK-NEXT: Results
// CHECK-NEXT: 1000
// CHECK-NEXT: 1100
