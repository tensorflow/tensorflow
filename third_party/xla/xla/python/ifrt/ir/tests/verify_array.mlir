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
  // expected-error@+2 {{Requires the same amount of `devices` and from `sharding`. Actual: 3 vs 4}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<2x2 to [0,1] on 2x2>,
                  [0,1,2]>
  return
}

// -----

func.func @array_requires_rank_matching_dim_shards() {
  // expected-error@+2 {{Requires dim shards to have the same rank as the array. Array rank is 2 vs dim shards rank of 0}}
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
