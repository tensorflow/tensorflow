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
// RUN: not mlir-interpreter-runner %s -run-all 2>&1 | FileCheck %s

func.func @write_4_at_3_inbounds() {
  %a = memref.alloc() : memref<5xi32>
  %base = arith.constant 3 : index
  %f = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  vector.transfer_write %f, %a[%base]
    {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]}
    : vector<4xi32>, memref<5xi32>
  return
}

// CHECK-LABEL: @write_4_at_3_inbounds
// CHECK-NEXT: index out of bounds

func.func @transfer_read_2d_1d_oob()-> vector<2xi32> {
  %a = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : memref<2x4xi32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c-42 = arith.constant -42: i32
  %f = vector.transfer_read %a[%c2, %c0], %c-42
      : memref<2x4xi32>, vector<2xi32>
  return %f : vector<2xi32>
}

// CHECK-LABEL: @transfer_read_2d_1d_oob
// CHECK-NEXT: index out of bounds

func.func @store_vector_memref() -> memref<1x2xvector<2xi32>> {
  %m = memref.alloc() : memref<1x2xvector<2xi32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1094795585 : index
  %v = arith.constant dense<[1,2]> : vector<2xi32>
  vector.store %v, %m[%c0, %c1] : memref<1x2xvector<2xi32>>, vector<2xi32>
  return %m : memref<1x2xvector<2xi32>>
}

// CHECK-LABEL: @store_vector_memref
// CHECK-NEXT: index out of bounds
