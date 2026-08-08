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
// RUN: xla-opt %s --triton-xla-atomics | FileCheck %s

// Test lowering of AtomicSpinWaitOp to tt.extern_elementwise

// CHECK-LABEL: tt.func @nomask_kernel
// CHECK-SAME:    %[[ARG0:.*]]: !tt.ptr<i32>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @nomask_kernel(%ptr : !tt.ptr<i32>, %expected : i32) {
// CHECK-NEXT:  %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG1]]
// CHECK-SAME:  {libname = "", libpath = "", pure = false, symbol = "xla_atomicspinwait_relaxed_gpu_eq_nomask"}
// CHECK-SAME:  : (!tt.ptr<i32>, i32) -> i32
  triton_xla.atomic_spin_wait gpu, relaxed, %ptr, equal_to, %expected  : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @masked_kernel
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<4xi1>
// CHECK-SAME:    %[[ARG2:.*]]: i32
tt.func @masked_kernel(
  %ptr: tensor<4x!tt.ptr<i32>>,
  %mask: tensor<4xi1>,
  %expected: i32
) {
// CHECK:         %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG2]], %[[ARG1]]
// CHECK-SAME:    {libname = "", libpath = "", pure = false, symbol = "xla_atomicspinwait_acquire_gpu_lt_mask"}
// CHECK-SAME:    : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> tensor<4xi32>
  triton_xla.atomic_spin_wait gpu, acquire, %ptr, less_than, %expected, %mask
      : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @test_vectorized_lt
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @test_vectorized_lt(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: tt.extern_elementwise %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: symbol = "xla_atomicspinwait_acquire_system_lt_nomask"
  triton_xla.atomic_spin_wait sys, acquire, %ptr, less_than, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @test_vectorized_eq
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @test_vectorized_eq(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: tt.extern_elementwise %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: symbol = "xla_atomicspinwait_acquire_system_eq_nomask"
  triton_xla.atomic_spin_wait sys, acquire, %ptr, equal_to, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}
