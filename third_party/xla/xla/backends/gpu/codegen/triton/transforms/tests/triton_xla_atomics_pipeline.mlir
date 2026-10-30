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
// RUN: xla-opt %s --split-input-file --triton-xla-pipeline='target=10.0 num-warps=1' \
// RUN:   | FileCheck %s

// With one warp and 64 ranks, every thread owns two signal pointers. The scalar
// value/expected operand of the lowered atomics has to be splat to the pointer
// shape; otherwise the LLVM lowering of tt.extern_elementwise fails with "size
// mismatch when packing elements for LLVM struct".

// CHECK-LABEL: llvm.func @block_barrier_64_ranks
// CHECK-COUNT-2: llvm.store {{.*}} atomic release <alignment = 4> : i32, !llvm.ptr<1>
// CHECK-COUNT-2: llvm.load {{.*}} atomic acquire <alignment = 4> : !llvm.ptr<1> -> i32
tt.func @block_barrier_64_ranks(
  %ptr : !tt.ptr<i64>, %signal_value : i32, %rank : i32) {
  triton_xla.block_barrier %ptr, %rank, %signal_value <world_size = 64> :
    (!tt.ptr<i64>, i32, i32) -> ()
  tt.return
}

// -----

// CHECK-LABEL: llvm.func @atomic_write_64_ranks
// CHECK-COUNT-2: llvm.store {{.*}} atomic release <alignment = 4> : i32, !llvm.ptr<1>
tt.func @atomic_write_64_ranks(%base : !tt.ptr<i32>, %value : i32, %mask_scalar : i1) {
  %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
  %ptr = tt.addptr %splat, %range : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
  %mask = tt.splat %mask_scalar : i1 -> tensor<64xi1>
  triton_xla.atomic_write sys, release, %ptr, %value, %mask :
    (tensor<64x!tt.ptr<i32>>, i32, tensor<64xi1>) -> ()
  tt.return
}

// -----

// CHECK-LABEL: llvm.func @atomic_spin_wait_64_ranks
// CHECK-COUNT-2: llvm.load {{.*}} atomic acquire <alignment = 4> : !llvm.ptr<1> -> i32
tt.func @atomic_spin_wait_64_ranks(%base : !tt.ptr<i32>, %expected : i32, %mask_scalar : i1) {
  %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
  %ptr = tt.addptr %splat, %range : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
  %mask = tt.splat %mask_scalar : i1 -> tensor<64xi1>
  triton_xla.atomic_spin_wait sys, acquire, %ptr, greater_than_or_equal_to,
    %expected, %mask : (tensor<64x!tt.ptr<i32>>, i32, tensor<64xi1>) -> ()
  tt.return
}
