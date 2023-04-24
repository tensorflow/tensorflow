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
      !ifrt.array<tensor<4x6xi32>, 4x1 to [1,0,2] on 2x2x3, [0,1,2,3,4,5,6,7,8,9,10,11]>
  return
}

// -----

func.func @array_devices_should_be_distinct() {
  // expected-error@+2 {{`devices` has duplicated id 0}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, 1x1 to [0] on 2, [0,0]>
  return
}

// -----

func.func @array_requires_same_permutation_and_axis_sizes() {
  // expected-error@+2 {{Expect same size for `permutation` and `axis_sizes`. Actual 2 vs 1}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, 1x1 to [0,1] on 2, [0,1]>
  return
}

// -----

func.func @array_requires_enough_devices() {
  // expected-error@+2 {{Can't shard the dims 2, 2 to the mesh of 0 on 2}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, 2x2 to [0] on 2, [0,1]>
  return
}

// -----

func.func @array_requires_shard_distributable_to_axes() {
  // expected-error@+2 {{Dimension #1 of 2 shards can't be assigned to the axes}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 3, [0,1,2]>
  return
}

// -----

func.func @array_requires_same_size_of_devices_and_from_axes() {
  // expected-error@+2 {{Requires the same amount of `devices` and from `sharding`. Actual: 3 vs 4}}
  %0 = builtin.unrealized_conversion_cast to
      !ifrt.array<tensor<4x4xi32>, 2x2 to [0,1] on 2x2, [0,1,2]>
  return
}
