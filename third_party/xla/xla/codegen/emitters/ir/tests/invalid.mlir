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
// RUN: emitters_opt  %s -split-input-file -verify-diagnostics

#empty_map = #xla.indexing_map<"() -> ()">
func.func @empty_indexing_map() {
  // expected-error @+1 {{IndexingMap is empty}}
  xla.apply_indexing #empty_map
  func.return
}

// -----

#map0 = #xla.indexing_map<"(d0, d1)[s0] -> (d0, d1 + s0), domain: d0 in [1, 2], d1 in [5, 8], s0 in [0, 32]">
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  // expected-error @+1 {{operand count 1 does not match the sum of dimensions 2 and symbols 1 in the symbolic map}}
  %0:2 = xla.apply_indexing #map0 (%d0)
  func.return %0#0, %0#1 : index, index
}

// -----

#map0 = #xla.indexing_map<"(d0, d1) -> (d0, d1), domain: d0 in [0, 7], d1 in [0, 7]">
func.func @apply_indexing_multi_result_xla_range(%d0: index, %d1: index)
    -> (index, index) {
  // expected-error @+1 {{'xla.range' is only supported on single-result ops, but the op has 2 results}}
  %0:2 = xla.apply_indexing #map0 (%d0, %d1) {xla.range = [0 : index, 7 : index]}
  func.return %0#0, %0#1 : index, index
}

// -----

#map = #xla.indexing_map<"()[s0, s1] -> (s0, s1), domain: s0 in [0, 1024], s1 in [0, 32]">
func.func @loop_result_num_mismatch(%input: tensor<1024x32xf32>,
    %init: f32) -> (f32) {
  // expected-error @+1 {{mismatch in number of loop-carried values and results}}
   %sum:2 = "xla.loop"(%init) <{
      indexing_map_attr = #map,
      operandSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%i: index, %j: index, %r0: index, %r1: index, %iter: f32):
      %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
      %add = arith.addf %iter, %t : f32
      xla.yield %add : f32
    }) : (f32) -> (f32, f32)
  func.return %sum#0 : f32
}

// -----

#map = #xla.indexing_map<"()[s0] -> (s0, s0), domain: s0 in [0, 1024]">
func.func @loop_iv_num_mismatch(%input: tensor<1024x32xf32>,
    %init: f32) -> (f32) {
  // expected-error @+1 {{mismatch in number of induction variables 2 and RangeVars}}
   %sum = "xla.loop"(%init) <{
      indexing_map_attr = #map,
      operandSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%i: index, %j: index, %r0: index, %r1: index, %iter: f32):
      %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
      %add = arith.addf %iter, %t : f32
      xla.yield %add : f32
    }) : (f32) -> (f32)
  func.return %sum : f32
}

// -----

#map = #xla.indexing_map<"()[s0, s1] -> (s0, s1), domain: s0 in [0, 1024], s1 in [0, 32]">

func.func @loop_types_mismatch(%input: tensor<1024x32xf32>, %init: f32) -> (i32) {
  // expected-error @+1 {{block iter arg type = 'f32', result type = 'i32' and init operand type = 'f32' should match}}
   %sum = "xla.loop"(%init) <{
      indexing_map_attr = #map,
      operandSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%i: index, %j: index, %r0: index, %r1: index, %iter: f32):
      %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
      %add = arith.addf %iter, %t : f32
      xla.yield %add : f32
    }) : (f32) -> (i32)
  func.return %sum : i32
}

// -----

#map = #xla.indexing_map<"(d0)[s0, s1] -> (s0, s1), domain: d0 in [0, 3], s0 in [0, 1024], s1 in [0, 32]">

func.func @loop_op(%input: tensor<1024x32xf32>, %init: f32, %dim: index) -> (f32) {
  // expected-error @+1 {{mismatch in number of dims operands 0 and DimVars in the indexing map}}
  %sum = xla.loop ()[%i, %j] -> (%r0, %r1)
     in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  } {xla.range = [0 : index, 42 : index]}
  func.return %sum : f32
}

// -----

func.func @atomic_store_indices_rank_mismatch(
    %in: tensor<2x3xf32>, %val: f32, %i: index) -> tensor<2x3xf32> {
  // expected-error @+1 {{indices count must match dest rank}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<2x3xf32>
  return %ret : tensor<2x3xf32>
}

// -----

func.func @atomic_store_value_type_mismatch(
    %in: tensor<2x3xf32>, %val: i32, %i: index, %j: index) -> tensor<2x3xf32> {
  // expected-error @+1 {{failed to verify that value type matches dest element type}}
  %ret = "xla.atomic_store"(%val, %in, %i, %j) : (i32, tensor<2x3xf32>, index, index) -> tensor<2x3xf32>
  return %ret : tensor<2x3xf32>
}

// -----

func.func @atomic_store_result_type_mismatch(
    %in: tensor<2x3xf32>, %val: f32, %i: index, %j: index) -> tensor<3x2xf32> {
  // expected-error @+1 {{failed to verify that result type matches dest type}}
  %ret = "xla.atomic_store"(%val, %in, %i, %j) : (f32, tensor<2x3xf32>, index, index) -> tensor<3x2xf32>
  return %ret : tensor<3x2xf32>
}

// -----

func.func @atomic_store_dynamic_shape(
    %in: tensor<?xf32>, %val: f32, %i: index) -> tensor<?xf32> {
  // expected-error @+1 {{'dest' must be statically shaped tensor}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<?xf32>
  return %ret : tensor<?xf32>
}

// -----

func.func @atomic_store_unsupported_element_type(
    %in: tensor<4xindex>, %val: index, %i: index) -> tensor<4xindex> {
  // expected-error @+1 {{expected signless integer, float, or complex float element type}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<4xindex>
  return %ret : tensor<4xindex>
}

// -----

func.func @atomic_rmw_zero_block_args(
    %in: tensor<2xf32>, %i: index) -> tensor<2xf32> {
  // expected-error @+1 {{expected body block to have a single argument}}
  %ret = xla.atomic_rmw %in[%i] : tensor<2xf32> {
  ^bb0:
    %c = arith.constant 1.0 : f32
    xla.yield %c : f32
  }
  return %ret : tensor<2xf32>
}

// -----

func.func @atomic_rmw_two_block_args(
    %in: tensor<2xf32>, %i: index) -> tensor<2xf32> {
  // expected-error @+1 {{expected body block to have a single argument}}
  %ret = xla.atomic_rmw %in[%i] : tensor<2xf32> {
  ^bb0(%a: f32, %b: i32):
    xla.yield %a : f32
  }
  return %ret : tensor<2xf32>
}

// -----

func.func @atomic_rmw_indices_rank_mismatch(
    %in: tensor<2xf32>, %i: index, %j: index) -> tensor<2xf32> {
  // expected-error @+1 {{indices count must match input rank}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2xf32> {
  ^bb0(%a: f32):
    xla.yield %a : f32
  }
  return %ret : tensor<2xf32>
}

// -----

func.func @atomic_rmw_scalable_vector(
    %in: tensor<16xf32>, %i: index) -> tensor<16xf32> {
  // expected-error @+1 {{expected a fixed length 1D vector type for the block argument}}
  %ret = xla.atomic_rmw %in[%i] : tensor<16xf32> {
  ^bb0(%a: vector<[2]xf32>):
    xla.yield %a : vector<[2]xf32>
  }
  return %ret : tensor<16xf32>
}

// -----

func.func @atomic_rmw_vector_on_0d_tensor(
    %in: tensor<f32>, %v: vector<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{vector block argument exceeds input tensor minor dimension}}
  %ret = xla.atomic_rmw %in[] : tensor<f32> {
  ^bb0(%a: vector<2xf32>):
    xla.yield %v : vector<2xf32>
  }
  return %ret : tensor<f32>
}

// -----

func.func @atomic_rmw_vector_on_zero_elements(
    %in: tensor<0x4xf32>, %i: index, %j: index, %v: vector<2xf32>) -> tensor<0x4xf32> {
  // expected-error @+1 {{vector block argument exceeds input tensor minor dimension}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<0x4xf32> {
  ^bb0(%a: vector<2xf32>):
    xla.yield %v : vector<2xf32>
  }
  return %ret : tensor<0x4xf32>
}

// -----

func.func @atomic_rmw_vector_layout_minor_dim_too_small(
    %in: tensor<1x4xf32, dense<[0, 1]> : tensor<2xi64>>, %i: index, %j: index,
    %v: vector<2xf32>) -> tensor<1x4xf32, dense<[0, 1]> : tensor<2xi64>> {
  // expected-error @+1 {{vector block argument exceeds input tensor minor dimension}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<1x4xf32, dense<[0, 1]> : tensor<2xi64>> {
  ^bb0(%a: vector<2xf32>):
    xla.yield %v : vector<2xf32>
  }
  return %ret : tensor<1x4xf32, dense<[0, 1]> : tensor<2xi64>>
}

// -----

func.func @atomic_store_tf32(
    %in: tensor<4xtf32>, %val: tf32, %i: index) -> tensor<4xtf32> {
  // expected-error @+1 {{expected signless integer, float, or complex float element type}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<4xtf32>
  return %ret : tensor<4xtf32>
}

// -----

func.func @atomic_store_non_power_of_two_int(
    %in: tensor<4xi24>, %val: i24, %i: index) -> tensor<4xi24> {
  // expected-error @+1 {{expected signless integer, float, or complex float element type}}
  %ret = xla.atomic_store %val into %in[%i] : tensor<4xi24>
  return %ret : tensor<4xi24>
}

// -----

func.func @atomic_rmw_malformed_layout_encoding(
    %in: tensor<4x1xf32, dense<[18446744073709551616, 1]> : tensor<2xi128>>,
    %i: index, %j: index, %v: vector<2xf32>)
    -> tensor<4x1xf32, dense<[18446744073709551616, 1]> : tensor<2xi128>> {
  // expected-error @+1 {{expected tensor layout encoding to be a 1D i64 minor to major permutation matching tensor rank}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<4x1xf32, dense<[18446744073709551616, 1]> : tensor<2xi128>> {
  ^bb0(%a: vector<2xf32>):
    xla.yield %v : vector<2xf32>
  }
  return %ret : tensor<4x1xf32, dense<[18446744073709551616, 1]> : tensor<2xi128>>
}

// -----

func.func @atomic_rmw_layout_encoding_repeats_dim(
    %in: tensor<4x4x4xf32, dense<[0, 2, 2]> : tensor<3xi64>>, %i: index,
    %j: index, %k: index)
    -> tensor<4x4x4xf32, dense<[0, 2, 2]> : tensor<3xi64>> {
  // expected-error @+1 {{expected tensor layout encoding to be a permutation of [0, rank)}}
  %ret = xla.atomic_rmw %in[%i, %j, %k] : tensor<4x4x4xf32, dense<[0, 2, 2]> : tensor<3xi64>> {
  ^bb0(%a: f32):
    xla.yield %a : f32
  }
  return %ret : tensor<4x4x4xf32, dense<[0, 2, 2]> : tensor<3xi64>>
}

// -----

func.func @atomic_store_layout_encoding_dim_out_of_range(
    %in: tensor<4x4xf32, dense<[2, 0]> : tensor<2xi64>>, %val: f32, %i: index,
    %j: index) -> tensor<4x4xf32, dense<[2, 0]> : tensor<2xi64>> {
  // expected-error @+1 {{expected tensor layout encoding to be a permutation of [0, rank)}}
  %ret = xla.atomic_store %val into %in[%i, %j] : tensor<4x4xf32, dense<[2, 0]> : tensor<2xi64>>
  return %ret : tensor<4x4xf32, dense<[2, 0]> : tensor<2xi64>>
}

// -----

func.func @atomic_rmw_overflow_element_count(
    %in: tensor<4611686018427387904x4xf32>, %i: index, %j: index)
    -> tensor<4611686018427387904x4xf32> {
  // expected-error @+1 {{input tensor element count overflows int64}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<4611686018427387904x4xf32> {
  ^bb0(%a: f32):
    xla.yield %a : f32
  }
  return %ret : tensor<4611686018427387904x4xf32>
}

// -----

func.func @atomic_rmw_mismatch_yield(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  // expected-error @+1 {{mismatch in the block argument type and the yielded type}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 420 : i32
      xla.yield %c42 : i32
  }
  return %ret : tensor<2x3xf32>
}

// -----

func.func @atomic_rmw_mismatch_block_arg_type(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  // expected-error @+1 {{mismatch in the result element type and the block argument type}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : i32):
      %c42 = arith.constant 420 : i32
      xla.yield %c42 : i32
  }
  return %ret : tensor<2x3xf32>
}

// -----

func.func @atomic_rmw_mismatch_block_arg_vector_type(%in: tensor<16xf32>, %i: index) -> (tensor<16xf32>) {
  // expected-error @+1 {{mismatch in the result element type and the block argument vector element type}}
  %ret = xla.atomic_rmw %in[%i] : tensor<16xf32> {
    ^bb0(%current : vector<2xi32>):
      %c42 = arith.constant dense<42> : vector<2xi32>
      %add = arith.addi %current, %c42 : vector<2xi32>
      xla.yield %add : vector<2xi32>
  }
  return %ret : tensor<16xf32>
}

// -----

// expected-error @+2 {{expected ::xla::BackendKind to be one of: cpu, gpu, tpu}}
// expected-error @+1 {{failed to parse XLA_BackendKindAttr parameter 'value' which is to be a `::xla::BackendKind`}}
func.func @test_backend_kind(%arg0: f32) attributes { xla.backend_kind = #xla.backend_kind<foo> } {
  func.return
}

// -----

func.func @copy_slice_element_type_mismatch(
    %src: tensor<4xf32>, %dst: tensor<8xi32>, %idx: index) -> tensor<8xi32> {
  // expected-error @+1 {{src and dest element types must match}}
  %out = xla.copy_slice %src[%idx] into %dst[%idx] count(2)
      : tensor<4xf32> into tensor<8xi32>
  func.return %out : tensor<8xi32>
}

// -----

func.func @copy_slice_src_indices_rank_mismatch(
    %src: tensor<4x2xf32>, %dst: tensor<8xf32>, %idx: index) -> tensor<8xf32> {
  // expected-error @+1 {{src_indices count must match src rank}}
  %out = xla.copy_slice %src[%idx] into %dst[%idx] count(2)
      : tensor<4x2xf32> into tensor<8xf32>
  func.return %out : tensor<8xf32>
}

// -----

func.func @copy_slice_negative_count(
    %src: tensor<4xf32>, %dst: tensor<8xf32>, %idx: index) -> tensor<8xf32> {
  // expected-error @+1 {{num_elements must be non-negative}}
  %out = xla.copy_slice %src[%idx] into %dst[%idx] count(-1)
      : tensor<4xf32> into tensor<8xf32>
  func.return %out : tensor<8xf32>
}

// -----

func.func @linear_extract_rank0(%src: tensor<f32>, %idx: index) -> f32 {
  // expected-error @+1 {{tensor must have rank at least 1}}
  %out = xla.linear_extract %src[%idx] : tensor<f32>
  func.return %out : f32
}

// -----

func.func @linear_insert_rank0(%val: f32, %dst: tensor<f32>, %idx: index) -> tensor<f32> {
  // expected-error @+1 {{tensor must have rank at least 1}}
  %out = xla.linear_insert %val into %dst[%idx] : tensor<f32>
  func.return %out : tensor<f32>
}

// -----

func.func @get_dynamic_dim_size_negative_dim(%src: tensor<4xf32>) -> i32 {
  // expected-error @+1 {{dim must be non-negative}}
  %out = xla.get_dynamic_dim_size %src -1 : tensor<4xf32>
  func.return %out : i32
}

// -----

func.func @set_dynamic_dim_size_negative_dim(%dst: tensor<4xf32>, %size: i32) -> tensor<4xf32> {
  // expected-error @+1 {{dim must be non-negative}}
  %out = xla.set_dynamic_dim_size %size into %dst[-1] : tensor<4xf32>
  func.return %out : tensor<4xf32>
}

// -----

module {
  %c0 = arith.constant 0 : index
  %v = arith.constant 1.0 : f32
  // expected-error @+1 {{must reside in an SSACFG region}}
  %0 = xla.atomic_store %v into %0[%c0] : tensor<4xf32>
}
