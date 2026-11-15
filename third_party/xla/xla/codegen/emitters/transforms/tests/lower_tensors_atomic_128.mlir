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
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 9}'" \
// RUN: | FileCheck %s

// Hopper has a 128 bit atomicCAS, which needs a 16 byte aligned address.

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
// CHECK:         %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i128
// CHECK:         scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK:           %[[NEW:.*]] = arith.muli %[[OLD]], %{{.*}} : i128
// CHECK:           llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128

// -----

func.func @atomic_store_c128(
    %in: tensor<8xcomplex<f64>> {llvm.align = 16 : index}, %val: complex<f64>,
    %i: index) -> tensor<8xcomplex<f64>> {
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f64>>
  return %ret : tensor<8xcomplex<f64>>
}
// CHECK-LABEL: @atomic_store_c128
// CHECK:         %[[ADDR:.*]] = llvm.getelementptr inbounds
// CHECK:         %[[VEC:.*]] = vector.from_elements %{{.*}}, %{{.*}} : vector<2xf64>
// CHECK:         %[[NEW:.*]] = llvm.bitcast %[[VEC]] : vector<2xf64> to i128
// CHECK:         %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i128
// CHECK:         scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK:           llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128

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
// CHECK:         %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i128
// CHECK:         scf.while (%[[OLD:.*]] = %[[INIT]]) : (i128) -> i128
// CHECK:           %[[OLD_VEC:.*]] = llvm.bitcast %[[OLD]] : i128 to vector<2xf64>
// CHECK:           %[[SUM:.*]] = complex.add
// CHECK:           %[[NEW:.*]] = llvm.bitcast %{{.*}} : vector<2xf64> to i128
// CHECK:           llvm.cmpxchg %[[ADDR]], %[[OLD]], %[[NEW]] monotonic monotonic : !llvm.ptr, i128

// -----

// complex<f64> only has the 8 byte ABI alignment of f64.
func.func @atomic_store_c128_abi_aligned(
    %in: tensor<8xcomplex<f64>> {llvm.align = 8 : index}, %val: complex<f64>,
    %i: index) -> tensor<8xcomplex<f64>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f64>>
  return %ret : tensor<8xcomplex<f64>>
}

// -----

func.func @atomic_rmw_c128_unknown_alignment(
    %in: tensor<8xcomplex<f64>>, %i: index) -> tensor<8xcomplex<f64>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_rmw %in[%i] : tensor<8xcomplex<f64>> {
    ^bb0(%current : complex<f64>):
      %a = complex.add %current, %current : complex<f64>
      xla.yield %a : complex<f64>
  }
  return %ret : tensor<8xcomplex<f64>>
}

// -----

func.func @atomic_rmw_i128_unknown_alignment(
    %in: tensor<8xi128>, %val: i128, %i: index) -> tensor<8xi128> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi128> {
    ^bb0(%current : i128):
      %m = arith.muli %current, %val : i128
      xla.yield %m : i128
  }
  return %ret : tensor<8xi128>
}

// -----

// Base alignment comes from the shared allocation element type (4 bytes for
// i8), not the reinterpreted f128 view.
func.func @atomic_store_f128_on_narrow_shared(%val: f128) -> tensor<1xf128> {
  %c0 = arith.constant 0 : index
  %shared = xla_gpu.allocate_shared : tensor<16xi8>
  %wide = builtin.unrealized_conversion_cast %shared : tensor<16xi8> to tensor<1xf128>
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %wide[%c0] : tensor<1xf128>
  return %ret : tensor<1xf128>
}
