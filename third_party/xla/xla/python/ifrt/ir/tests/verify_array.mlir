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
// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_array() {
  /// Dim 0 of the tensor is sharded into 4 slices.
  /// Dim 1 is unsharded.
  /// The 4 slices are distributed to the axes [1,0] of the 2x2x3 mesh.
  /// Axes 2 of size 3 is replicated.
  /// Specifically, the 4 slices are distributed to:
  ///   Slice 0 to device 0,4,8
  ///   Slice 1 to device 2,6,10
  ///   Slice 2 to device 1,5,9
  ///   Slice 3 to device 3,7,11
  /// The equivalent HloSharding is
  ///   {devices=[4,1,3]0,2,1,3,4,6,5,7,8,10,9,11 replicate_on_last_dim}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>,
                  #ifrt.sharding_param<4x1 to [1,0,2] on 2x2x3>,
                  [0,1,2,3,4,5,6,7,8,9,10,11]>
  return
}

#devices = #ifrt<devices[0,1,2,3]>
func.func @good_array_with_aliased_devices() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  #devices>
  return
}

// -----

func.func @good_array_scalar() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<i32>,#ifrt.sharding_param< to [0,1] on 2x2>, [0,1,2,3]>
  return
}

// -----

func.func @good_array_with_memory_kind_and_layout() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], memory_kind = "default", layout = "default">
  return
}

// -----

func.func @good_array_with_unreduced_axis() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<6xi32>, #ifrt.sharding_param<2 to [0,1] on 2x2
                  unreduced [1]>, [0,1,2,3]>
  return
}

// -----

func.func @good_array_with_multiple_unreduced_axes() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<2x1 to [1,2,0] on 3x2x2
                  unreduced [1,2]>, [0,1,2,3,4,5,6,7,8,9,10,11]>
  return
}

// -----

func.func @array_with_empty_memory_kind() {
  // expected-error@+4 {{Memory kind cannot be empty}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], memory_kind = "">
  return
}

// -----

func.func @array_with_bad_layout() {
  // expected-error@+4 {{Invalid layout mode}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "bad_layout">
  return
}

// -----

func.func @array_devices_should_be_distinct() {
  // expected-error@+3 {{Device list has duplicate logical id 0}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'devices_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0,0]>
  return
}

// -----

func.func @array_devices_should_be_non_negative() {
  // expected-error@+4 {{Device list has negative logical id -1}}
  // expected-error@+3 {{failed to parse Ifrt_ArrayType parameter 'devices_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                  [-1,0]>
  return
}

// -----

func.func @array_requires_same_permutation_and_axis_sizes() {
  // expected-error@+3 {{Expect same non-zero size for `permutation` and `axis_sizes`. Actual 2 vs 1}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x1 to [0,1] on 2>,
                  [0,1]>
  return
}

// -----

func.func @array_requires_enough_devices() {
  // expected-error@+3 {{Can't shard the dims 2x2 to the mesh of [0] on 2}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<2x2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_requires_shard_distributable_to_axes() {
  // expected-error@+3 {{Can't shard the dims 1x2 to the mesh of [0] on 3}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 3>,
                  [0,1,2]>
  return
}

// -----

func.func @array_requires_same_size_of_devices_and_from_axes() {
  // expected-error@+4 {{Requires the same amount of `devices` and from `sharding`. Actual: 3 vs 4}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<2x2 to [0,1] on 2x2>,
                  [0,1,2]>
  return
}

// -----

func.func @array_requires_rank_matching_dim_shards() {
  // expected-error@+4 {{Requires dim shards to have the same rank as the array. Array rank is 2 vs dim shards rank of 0}}
  %0 = builtin.unrealized_conversion_cast to
       !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param< to [0,1] on 2x2>,
                   [0,1,2,3]>
  return
}

// -----

func.func @array_requires_non_empty_permutation() {
  // expected-error@+3 {{Expect same non-zero size for `permutation` and `axis_sizes`. Actual 0 vs 0}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
       !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<2x2 to [] on>,
                   [0,1,2,3]>
  return
}

// -----

func.func @array_requires_unique_unreduced_axes() {
  // expected-error@+3 {{`unreduced_axes` has duplicate value: 0}}
  // expected-error@+3 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
       !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0,1] on 2x2
                    unreduced [0,0]>, [0,1,2,3]>
  return
}

// -----

func.func @array_requires_unreduced_axis_in_bounds() {
  // expected-error@+3 {{`unreduced_axes` must contain non-negative values less than the number of mesh dimensions (=3). Saw: 3}}
  // expected-error@+3 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<6xi32>, #ifrt.sharding_param<2 to [0,2,1] on 2x1x2
                  unreduced [3]>, [0,1,2,3]>
  return
}

// -----

func.func @good_array_token() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0,1] on 2x2>, [0,1,2,3]>
  return
}

// -----

func.func @array_token_cannot_have_shape() {
  // expected-error@+3 {{Token type cannot have non-empty shape}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x!ifrt.token>, #ifrt.sharding_param< to [0,1] on 2x2>, [0,1,2,3]>
  return
}

// -----

func.func @array_token_cannot_be_sharded() {
  // expected-error@+3 {{Requires dim shards to have the same rank as the array}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param<2 to [0,1] on 2x2>, [0,1,2,3]>
  return
}

// -----

func.func @array_devices_cannot_be_empty() {
  // expected-error@+4 {{expected integer value}}
  // expected-error@+3 {{failed to parse Ifrt_DevicesAttr parameter 'ids'}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'devices_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_unspecified, []>
  return
}

// -----

func.func @array_axis_sizes_must_be_positive() {
  // expected-error@+3 {{`axis_sizes` must be positive, got 0 in [0]}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x1 to [0] on 0>, [0]>
  return
}

// -----

func.func @array_axis_sizes_cannot_overflow_int32() {
  // expected-error@+4 {{`axis_sizes` value 4294967297 overflows 32-bit signed integer}}
  // expected-error@+3 {{failed to parse Ifrt_ShardingParamAttr parameter 'sharding'}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x1 to [0] on 4294967297>, [0]>
  return
}

// -----

func.func @array_dim_shards_cannot_overflow_int32() {
  // expected-error@+3 {{`dim_shards` product overflows}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<4294967297x1 to [0] on 1>, [0]>
  return
}

// -----

func.func @array_shape_cannot_have_encoding() {
  // expected-error@+3 {{Unexpected tensor encoding on array shape}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32, "enc">, #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
  return
}

// -----

func.func @array_unsupported_element_type() {
  // expected-error@+3 {{Unsupported array element type: '!ifrt.control'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x!ifrt.control>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_shape_element_count_overflow() {
  // expected-error@+3 {{Array shape element count exceeds maximum supported elements (72057594037927935)}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<65536x281474976710656xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
  return
}

// -----

func.func @array_shape_exceeds_max_num_elements() {
  // expected-error@+3 {{Array shape element count exceeds maximum supported elements (72057594037927935)}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<72057594037927936xi8>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
  return
}

// -----

func.func @array_shape_zero_dim_exceeds_max_num_elements() {
  // expected-error@+3 {{Array shape element count exceeds maximum supported elements (72057594037927935)}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<0x72057594037927936xi8>, #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
  return
}

// -----

func.func @good_array_with_user_layout() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{1,0}">
  %1 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{0,1:T(8,128)}">
  return
}

// -----

func.func @good_array_scalar_with_empty_user_layout() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 1>, [0],
                  layout = "{}">
  return
}

// -----

func.func @good_array_token_with_default_layout() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 1>, [0],
                  layout = "default">
  return
}

// -----

func.func @good_array_string_with_default_layout() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x!ifrt.string>, #ifrt.sharding_param<2 to [0] on 2>,
                  [0,1], layout = "default">
  return
}

// -----

func.func @good_array_with_auto_layout() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x3x4xi32>,
                  #ifrt.sharding_param<1x1x1 to [0] on 1>, [0],
                  layout = "auto">
  return
}

// -----

func.func @array_user_layout_rank_too_small() {
  // expected-error@+5 {{Layout minor_to_major size (2) does not match array rank (3): "{0,1}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x3x4xi32>,
                  #ifrt.sharding_param<1x1x1 to [0] on 1>, [0],
                  layout = "{0,1}">
  return
}

// -----

func.func @array_user_layout_rank_too_large() {
  // expected-error@+4 {{Layout minor_to_major size (3) does not match array rank (2): "{0,1,2}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{0,1,2}">
  return
}

// -----

func.func @array_scalar_with_non_empty_user_layout() {
  // expected-error@+4 {{Layout minor_to_major size (1) does not match array rank (0): "{0}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 1>, [0],
                  layout = "{0}">
  return
}

// -----

func.func @array_token_with_empty_user_layout() {
  // expected-error@+4 {{Array with element type '!ifrt.token' only supports default layout mode, got "{}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 1>, [0],
                  layout = "{}">
  return
}

// -----

func.func @array_token_with_user_layout_modifiers() {
  // expected-error@+4 {{Array with element type '!ifrt.token' only supports default layout mode, got "{:T(0)E(-1)S(5)}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 1>, [0],
                  layout = "{:T(0)E(-1)S(5)}">
  return
}

// -----

func.func @array_token_with_auto_layout() {
  // expected-error@+4 {{Array with element type '!ifrt.token' only supports default layout mode, got "auto"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 1>, [0],
                  layout = "auto">
  return
}

// -----

func.func @array_string_with_user_layout() {
  // expected-error@+4 {{Array with element type '!ifrt.string' only supports default layout mode, got "{0:T(128)}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x!ifrt.string>, #ifrt.sharding_param<2 to [0] on 2>,
                  [0,1], layout = "{0:T(128)}">
  return
}

// -----

func.func @array_string_with_auto_layout() {
  // expected-error@+4 {{Array with element type '!ifrt.string' only supports default layout mode, got "auto"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x!ifrt.string>, #ifrt.sharding_param<2 to [0] on 2>,
                  [0,1], layout = "auto">
  return
}

// -----

func.func @array_user_layout_empty_minor_to_major_with_memory_space() {
  // expected-error@+4 {{Layout minor_to_major size (0) does not match array rank (2): "{:S(1)}"}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{:S(1)}">
  return
}

// -----

func.func @array_user_layout_out_of_bounds_dim() {
  // expected-error@+4 {{Invalid layout "{2,0:T(1,128)}": layout minor_to_major field has out-of-bounds value}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{2,0:T(1,128)}">
  return
}

// -----

func.func @array_user_layout_negative_dim() {
  // expected-error@+4 {{Invalid layout "{-1,0}": layout minor_to_major field has out-of-bounds value}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{-1,0}">
  return
}

// -----

func.func @array_user_layout_duplicate_dim() {
  // expected-error@+4 {{Invalid layout "{0,0}": layout minor_to_major field has duplicate values}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{0,0}">
  return
}

// -----

func.func @array_user_layout_zero_tile_dim() {
  // expected-error@+4 {{Invalid layout "{1,0:T(0,128)}": layout has invalid tiles}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{1,0:T(0,128)}">
  return
}

// -----

func.func @array_user_layout_empty_tile() {
  // expected-error@+4 {{Invalid layout "{1,0:T()}": layout has invalid tiles}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{1,0:T()}">
  return
}

// -----

func.func @array_user_layout_negative_element_size() {
  // expected-error@+4 {{Invalid layout "{1,0:E(-8)}": layout element_size_in_bits field is negative}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{1,0:E(-8)}">
  return
}

// -----

func.func @array_user_layout_oob_split_config_on_local_shape() {
  // expected-error@+4 {{Invalid layout "{1,0:SC(0:2)}": split config split index is out of bounds: 2 (dim size 1)}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_param<4x1 to [0,1] on 2x2>,
                  [0,1,2,3], layout = "{1,0:SC(0:2)}">
  return
}

// -----

func.func @good_array_user_layout_with_non_divisible_sharding() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<5x6xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                  [0,1], layout = "{1,0}">
  return
}

// -----

func.func @good_array_user_layout_with_unspecified_sharding() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x6xi32>, #ifrt.sharding_unspecified,
                  [0,1], layout = "{1,0}">
  return
}

// -----

func.func @good_array_string() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2x!ifrt.string>, #ifrt.sharding_param<2 to [0] on 2>,
                  [0,1]>
  return
}

// -----

func.func @good_array_xla_element_types() {
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xi1>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  %1 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xui32>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  %2 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xbf16>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  %3 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xf8E4M3FN>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  %4 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xcomplex<f32>>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  %5 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xcomplex<f64>>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_unsupported_index_element_type() {
  // expected-error@+3 {{Unsupported array element type: 'index'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xindex>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_unsupported_vector_element_type() {
  // expected-error@+3 {{Unsupported array element type: 'vector<2xi32>'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xvector<2xi32>>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_unsupported_i128_element_type() {
  // expected-error@+3 {{Unsupported array element type: 'i128'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xi128>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_unsupported_f128_element_type() {
  // expected-error@+3 {{Unsupported array element type: 'f128'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xf128>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @array_unsupported_complex_i32_element_type() {
  // expected-error@+3 {{Unsupported array element type: 'complex<i32>'}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<2xcomplex<i32>>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
  return
}
