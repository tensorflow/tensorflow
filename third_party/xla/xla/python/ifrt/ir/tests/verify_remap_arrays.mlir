// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @remap_from_two_to_one_array(
    %arg0: !array0 {ifrt.donated}, %arg1: !array1 {ifrt.donated})
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      {donated=true}
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
  %0, %1, %ctrl_0 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:2:1] to [0:2:1]>]>,
                #ifrt.array_mapping<0, 1, [#ifrt.mapping<[2:4:1] to [0:2:1]>]>]
      : (!array) -> (!array0, !array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array3 = !ifrt.array<tensor<2x2xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
func.func @remap_arrays_of_different_dtypes(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  %0, %1, %ctrl_0 = ifrt.RemapArrays(%arg0, %arg1) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
       #ifrt.array_mapping<1, 1, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0, !array1) -> (!array2, !array3)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array3 = !ifrt.array<tensor<1x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
func.func @requires_same_per_shard_shape(
    %arg0: !array0, %arg1: !array1) attributes {ifrt.function} {
  %0, %1, %ctrl_0 = ifrt.RemapArrays(%arg0, %arg1) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
       #ifrt.array_mapping<1, 1, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0, !array1) -> (!array2, !array3)
  return
}


// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_at_least_one_input() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires at least one input array}}
  %0, %ctrl_0 = ifrt.RemapArrays() mappings = [] : () -> (!array)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_at_least_one_mappping(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires at least one mapping}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings = [] : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
func.func @requires_mapped_arrays_have_same_dtype(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.RemapArrays' op requires input array #0 and output array #0 to have the same dtype}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<4x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @requires_same_per_shard_shape(
    %arg0: !array0) attributes {ifrt.function} {
  // expected-error@+2 {{'ifrt.RemapArrays' op Arrays have different per-shard shapes:}}
  // expected-error@+1 {{'ifrt.RemapArrays' op requires input array #0 and output array #0 to have the same per-shard shape}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0) -> (!array1)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_negative_start(%arg0: !array) attributes {ifrt.function} {
  // expected-error@+4 {{start, end must be zero or positive}}
  // expected-error@+3 {{failed to parse Ifrt_MappingAttr parameter}}
  // expected-error@+2 {{failed to parse Ifrt_ArrayMappingAttr parameter}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
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
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
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
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
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
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
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
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
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
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
    mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
    : (!array) -> (!array)
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @error_output_shard_has_more_than_one_input_shard_mapped(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{op output array #0 shard #0 is already assigned}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
    mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<0, 0, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
    : (!array) -> (!array)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1], layout = "{0,1}">
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0], layout = "{1,0}">
func.func @error_different_layouts(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{op requires input array #0 and output array #0 to have the same layout}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1], layout = "auto">
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
func.func @error_auto_layout(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{op does not allow input or output arrays with `auto` layout}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1], memory_kind = "device">
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0], memory_kind = "pinned_host">
func.func @error_different_memory_kind(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{op requires input array #0 and output array #0 to have the same memory kind}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0) mappings =
      [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array0) -> (!array1)
  return
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
func.func @main(%arg0: !array0 {ifrt.donated}, %arg1: !array1 {ifrt.donated})
    -> !array attributes {ifrt.function} {
  // expected-error@+1 {{op all arguments must be donated because multiple input arrays are mapped to output array #0}}
  %0, %ctrl_0 = ifrt.RemapArrays(%arg0, %arg1)
    mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
              #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
    : (!array0, !array1) -> (!array)
  return %0 : !array
}
