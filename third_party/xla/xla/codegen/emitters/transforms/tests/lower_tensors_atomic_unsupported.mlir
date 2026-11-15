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
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: --verify-diagnostics -xla-lower-tensors="target_type=cpu"

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: --verify-diagnostics \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 8}'"

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: --verify-diagnostics \
// RUN: -xla-lower-tensors="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}'"

// A complex<f32> element only has the 4 byte ABI alignment of f32, so the 64
// bit atomic word that holds it needs a proven 8 byte alignment.

func.func @atomic_store_c64_unknown_alignment(
    %in: tensor<8xcomplex<f32>>, %val: complex<f32>, %i: index)
    -> tensor<8xcomplex<f32>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f32>>
  return %ret : tensor<8xcomplex<f32>>
}

// -----

func.func @atomic_store_c64_abi_aligned(
    %in: tensor<8xcomplex<f32>> {llvm.align = 4 : index}, %val: complex<f32>,
    %i: index) -> tensor<8xcomplex<f32>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f32>>
  return %ret : tensor<8xcomplex<f32>>
}

// -----

func.func @atomic_rmw_c64_unknown_alignment(
    %in: tensor<8xcomplex<f32>>, %i: index) -> tensor<8xcomplex<f32>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_rmw %in[%i] : tensor<8xcomplex<f32>> {
    ^bb0(%current : complex<f32>):
      %a = complex.add %current, %current : complex<f32>
      xla.yield %a : complex<f32>
  }
  return %ret : tensor<8xcomplex<f32>>
}

// -----

// These targets have no 128 bit atomicCAS that the lowering can rely on, even
// for a 16 byte aligned address.

func.func @atomic_rmw_i128(
    %in: tensor<8xi128> {llvm.align = 16 : index}, %val: i128, %i: index)
    -> tensor<8xi128> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi128> {
    ^bb0(%current : i128):
      %m = arith.muli %current, %val : i128
      xla.yield %m : i128
  }
  return %ret : tensor<8xi128>
}

// -----

func.func @atomic_store_f128(
    %in: tensor<8xf128> {llvm.align = 16 : index}, %val: f128, %i: index)
    -> tensor<8xf128> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xf128>
  return %ret : tensor<8xf128>
}

// -----

func.func @atomic_store_c128(
    %in: tensor<8xcomplex<f64>> {llvm.align = 16 : index}, %val: complex<f64>,
    %i: index) -> tensor<8xcomplex<f64>> {
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<8xcomplex<f64>>
  return %ret : tensor<8xcomplex<f64>>
}

// -----

// No buffer backs the result of an unknown op, so there is no address to store
// to.

func.func @atomic_store_untraceable_destination(%val: f32, %i: index)
    -> tensor<8xf32> {
  %unknown = "dummy.op"() : () -> tensor<8xf32>
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %unknown[%i] : tensor<8xf32>
  return %ret : tensor<8xf32>
}

// -----

func.func @atomic_rmw_untraceable_destination(%val: i32, %i: index)
    -> tensor<8xi32> {
  %unknown = "dummy.op"() : () -> tensor<8xi32>
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_rmw %unknown[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %a = arith.addi %current, %val : i32
      xla.yield %a : i32
  }
  return %ret : tensor<8xi32>
}
