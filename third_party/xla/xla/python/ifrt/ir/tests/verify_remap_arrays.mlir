// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @remap_from_two_to_one_array(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  %0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      : (!array0, !array1) -> (!array2)
  return
}

// -----

!array = !ifrt.array<tensor<2x8xi32>,
                     #ifrt.sharding_param<1x4 to [0] on 4>, [0,1,2,3]>
!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
func.func @remap_from_one_array_to_two_arrays(%arg0: !array)
    attributes {ifrt.function} {
  %0, %1 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:2:1] to [0:2:1]>]>,
                #ifrt.array_mapping<0, 1, [#ifrt.mapping<[2:4:1] to [0:2:1]>]>]
      : (!array) -> (!array0, !array1)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_at_least_one_input() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires at least one input array}}
  %0 = ifrt.RemapArrays() mappings = [] : () -> (!array)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_inputs_have_same_dtype(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires every input and output array to have the same dtype}}
  %0 = ifrt.RemapArrays(%arg0, %arg1) mappings = []
      : (!array0, !array1) -> (!array0)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_outputs_have_same_dtype(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires every input and output array to have the same dtype}}
  %0 = ifrt.RemapArrays(%arg0) mappings = []
      : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @requires_different_input_per_shard_shape(
    %arg0: !array0, %arg1: !array1) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires every input array to have the same per-shard shape}}
  %0 = ifrt.RemapArrays(%arg0, %arg1) mappings = []
      : (!array0, !array1) -> (!array0)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_different_output_per_shard_shape(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires every output array to have the same per-shard shape}}
  %0, %1 = ifrt.RemapArrays(%arg0) mappings = [] : (!array0) -> (!array0, !array1)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_negative_start(%arg0: !array) attributes {ifrt.function} {
  // expected-error@+4 {{start, end must be zero or positive}}
  // expected-error@+3 {{failed to parse Ifrt_MappingAttr parameter}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayMappingAttr parameter}}
  %0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[-1:1:1] to [1:2:1]>]>]
      : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_non_positive_step(%arg0: !array) attributes {ifrt.function} {
  // expected-error@+4 {{step must be positive}}
  // expected-error@+3 {{failed to parse Ifrt_MappingAttr parameter}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayMappingAttr parameter}}
  %0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [1:2:0]>]>]
      : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_empty_interval(%arg0: !array) attributes {ifrt.function} {
  // expected-error@+4 {{interval is empty}}
  // expected-error@+3 {{failed to parse Ifrt_MappingAttr parameter}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayMappingAttr parameter}}
  %0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[1:0:1] to [1:2:0]>]>]
      : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_diff_number_shards(%arg0: !array) attributes {ifrt.function} {
  // expected-error@+3 {{but they must have the same number of shards}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayMappingAttr parameter}}
  %0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:2:1]>]>]
      : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_input_shard_used_more_than_once(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{op input array #0 shard #0 is already used}}
  %0 = ifrt.RemapArrays(%arg0)
    mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
    : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_output_shard_has_no_input_shard_mapped(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{op output array #0 shard #1 is unassigned.}}
  %0 = ifrt.RemapArrays(%arg0)
    mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
    : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_output_shard_has_more_than_on_input_shard_mapped(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{op output array #0 shard #0 is already assigned}}
  %0 = ifrt.RemapArrays(%arg0)
    mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<0, 0, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
    : (!array) -> (!array)
  return
}
