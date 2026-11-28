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
// RUN: emitters_opt %s -split-input-file --verify-diagnostics \
// RUN: -xla-cpu-lower-tensors="has_atomic_cmpxchg_128=true" \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-CAS-STORE

// RUN: emitters_opt %s -split-input-file --verify-diagnostics \
// RUN: -xla-cpu-lower-tensors="has_atomic_cmpxchg_128=true has_atomic_load_store_128=true" \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-DIRECT-STORE

// Qualifying 64 bit CPUs support 128 bit atomics on 16 byte aligned addresses.
// Targets with native 128 bit monotonic atomic stores emit llvm.store directly;
// cmpxchg only targets fall back to a cmpxchg loop.

func.func @atomic_rmw_i128(
    %in: tensor<8xi128> {llvm.align = 16 : index}, %val: i128, %i: index)
    -> tensor<8xi128> {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi128> {
    ^bb0(%current : i128):
      %m = arith.muli %current, %val : i128
      xla.yield %m : i128
  }
  return %ret : tensor<8xi128>
}
// CHECK-LABEL: @atomic_rmw_i128
// CHECK:         %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:         %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK:         scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK:           %[[NEW:.*]] = arith.muli %[[OLD]], %{{.*}} : i128
// CHECK:           llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128

// -----

func.func @atomic_rmw_add_i128(
    %in: tensor<8xi128> {llvm.align = 16 : index}, %val: i128, %i: index)
    -> tensor<8xi128> {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi128> {
    ^bb0(%current : i128):
      %sum = arith.addi %current, %val : i128
      xla.yield %sum : i128
  }
  return %ret : tensor<8xi128>
}
// CHECK-LABEL: @atomic_rmw_add_i128
// CHECK-NOT:     llvm.atomicrmw
// CHECK:         %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:         %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK:         scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK:           %[[NEW:.*]] = arith.addi %[[OLD]], %{{.*}} : i128
// CHECK:           llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128

// -----

func.func @atomic_store_i128(
    %in: tensor<8xi128> {llvm.align = 16 : index}, %val: i128, %i: index)
    -> tensor<8xi128> {
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xi128>
  return %ret : tensor<8xi128>
}
// CHECK-LABEL:            @atomic_store_i128
// CHECK:                    %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK-CAS-STORE:          %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK-CAS-STORE:          scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK-CAS-STORE:            llvm.cmpxchg %[[ADDR]], %[[OLD]], %{{.*}} monotonic monotonic : !llvm.ptr, i128
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg
// CHECK-DIRECT-STORE:       llvm.store %{{.*}}, %[[ADDR]] atomic monotonic <alignment = 16> : i128, !llvm.ptr
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg

// -----

func.func @atomic_store_c128(
    %in: tensor<8xcomplex<f64>> {llvm.align = 16 : index}, %val: complex<f64>,
    %i: index) -> tensor<8xcomplex<f64>> {
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f64>>
  return %ret : tensor<8xcomplex<f64>>
}
// CHECK-LABEL:            @atomic_store_c128
// CHECK:                    %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:                    %[[VEC:.*]] = vector.from_elements %{{.*}}, %{{.*}} : vector<2xf64>
// CHECK:                    %[[NEW:.*]] = llvm.bitcast %[[VEC]] : vector<2xf64> to i128
// CHECK-CAS-STORE:          %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK-CAS-STORE:          scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK-CAS-STORE:            llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg
// CHECK-DIRECT-STORE:       llvm.store %[[NEW]], %[[ADDR]] atomic monotonic <alignment = 16> : i128, !llvm.ptr
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg

// -----

func.func @atomic_store_c128_overaligned(
    %in: tensor<8xcomplex<f64>> {llvm.align = 64 : index},
    %val: complex<f64>) -> tensor<8xcomplex<f64>> {
  %c4 = arith.constant 4 : index
  %ret = xla.atomic_store %val into %in[%c4] : tensor<8xcomplex<f64>>
  return %ret : tensor<8xcomplex<f64>>
}
// CHECK-LABEL:            @atomic_store_c128_overaligned
// CHECK:                    %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:                    %[[VEC:.*]] = vector.from_elements %{{.*}}, %{{.*}} : vector<2xf64>
// CHECK:                    %[[NEW:.*]] = llvm.bitcast %[[VEC]] : vector<2xf64> to i128
// CHECK-CAS-STORE:          %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK-CAS-STORE:          scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK-CAS-STORE:            llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg
// CHECK-DIRECT-STORE:       llvm.store %[[NEW]], %[[ADDR]] atomic monotonic <alignment = 64> : i128, !llvm.ptr
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg

// -----

func.func @atomic_store_f128(
    %in: tensor<8xf128> {llvm.align = 16 : index}, %val: f128, %i: index)
    -> tensor<8xf128> {
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xf128>
  return %ret : tensor<8xf128>
}
// CHECK-LABEL:            @atomic_store_f128
// CHECK:                    %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:                    %[[BITS:.*]] = arith.bitcast %{{.*}} : f128 to i128
// CHECK-CAS-STORE:          %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK-CAS-STORE:          scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK-CAS-STORE:            llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[BITS]] monotonic monotonic : !llvm.ptr, i128
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg
// CHECK-DIRECT-STORE:       llvm.store %[[BITS]], %[[ADDR]] atomic monotonic <alignment = 16> : i128, !llvm.ptr
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg

// -----

func.func @atomic_rmw_c128(
    %in: tensor<8xcomplex<f64>> {llvm.align = 16 : index}, %i: index)
    -> tensor<8xcomplex<f64>> {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xcomplex<f64>> {
    ^bb0(%current : complex<f64>):
      %a = complex.add %current, %current : complex<f64>
      xla.yield %a : complex<f64>
  }
  return %ret : tensor<8xcomplex<f64>>
}
// CHECK-LABEL: @atomic_rmw_c128
// CHECK:         %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:         %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK:         scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK:           %[[OLD_VEC:.*]] = llvm.bitcast %[[OLD]] : i128 to vector<2xf64>
// CHECK:           %[[SUM:.*]] = complex.add
// CHECK:           %[[NEW:.*]] = llvm.bitcast %{{.*}} : vector<2xf64> to i128
// CHECK:           llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128

// -----

func.func @atomic_rmw_overwrite_c128(
    %in: tensor<8xcomplex<f64>> {llvm.align = 16 : index}, %val: complex<f64>,
    %i: index) -> tensor<8xcomplex<f64>> {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xcomplex<f64>> {
    ^bb0(%current : complex<f64>):
      xla.yield %val : complex<f64>
  }
  return %ret : tensor<8xcomplex<f64>>
}
// CHECK-LABEL:            @atomic_rmw_overwrite_c128
// CHECK:                    %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK-CAS-STORE:          %[[INIT:.*]] = llvm.load %[[ADDR]] atomic monotonic <alignment = 16> : !llvm.ptr -> i128
// CHECK-CAS-STORE:          scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK-CAS-STORE:            llvm.cmpxchg %[[ADDR]], %[[OLD]], %{{.*}} monotonic monotonic : !llvm.ptr, i128
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg
// CHECK-DIRECT-STORE:       llvm.store %{{.*}}, %[[ADDR]] atomic monotonic <alignment = 16> : i128, !llvm.ptr
// CHECK-DIRECT-STORE-NOT:   llvm.cmpxchg

// -----

func.func @atomic_store_c128_abi_aligned(
    %in: tensor<8xcomplex<f64>> {llvm.align = 8 : index}, %val: complex<f64>,
    %i: index) -> tensor<8xcomplex<f64>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f64>>
  return %ret : tensor<8xcomplex<f64>>
}

// -----

func.func @atomic_rmw_c128_abi_aligned(
    %in: tensor<8xcomplex<f64>> {llvm.align = 8 : index}, %i: index)
    -> tensor<8xcomplex<f64>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_rmw %in[%i] : tensor<8xcomplex<f64>> {
    ^bb0(%current : complex<f64>):
      %a = complex.add %current, %current : complex<f64>
      xla.yield %a : complex<f64>
  }
  return %ret : tensor<8xcomplex<f64>>
}

// -----

func.func @atomic_store_i128_unknown_alignment(
    %in: tensor<8xi128>, %val: i128, %i: index) -> tensor<8xi128> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xi128>
  return %ret : tensor<8xi128>
}
