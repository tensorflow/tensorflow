// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
!array3 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
func.func @copy_two_different_arrays(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg1)
    : (!array0, !array1) -> (!array2, !array3)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
func.func @copy_donated_array(%arg0: !array0)
    attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) {donated=true}
    : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>, #ifrt.sharding_unspecified, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @copy_with_unspecified_input_sharding(%arg0: !array0)
    attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>, #ifrt.sharding_unspecified, [0,1]>
func.func @copy_with_unspecified_output_sharding(%arg0: !array0)
    attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_at_least_one_input() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires at least one input array}}
  %ctrl = ifrt.CopyArrays() : () -> ()
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_same_num_inputs_and_outputs(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires the same number of input and output arrays}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0) : (!array) -> (!array, !array)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_copied_array_to_have_same_dtype(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires input #1 and output #1 to have the same shape and dtype}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg0)
    : (!array0, !array0) -> (!array0, !array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<4x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_copied_array_to_have_same_shape(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires input #1 and output #1 to have the same shape and dtype}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg0)
    : (!array0, !array0) -> (!array0, !array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @requires_copied_array_to_have_same_sharding(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires input #0 and output #0 to have the same sharding}}
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [4,5]>
func.func @requires_inputs_with_same_devices(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires all input arrays to have the same devices}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg1)
    : (!array0, !array1) -> (!array2, !array2)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [4,5]>
func.func @requires_outputs_with_same_devices(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires all output arrays to have the same devices}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg0)
    : (!array0, !array0) -> (!array1, !array2)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1],
                      memory_kind = "device">
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1],
                      memory_kind = "host">
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>
func.func @requires_inputs_with_same_memory_kind(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires all input arrays to have the same memory kind}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg1)
    : (!array0, !array1) -> (!array2, !array2)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3],
                      memory_kind = "device">
!array2 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3],
                      memory_kind = "host">
func.func @requires_outputs_with_same_memory_kind(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op requires all output arrays to have the same memory kind}}
  %0, %1, %ctrl = ifrt.CopyArrays(%arg0, %arg0)
    : (!array0, !array0) -> (!array1, !array2)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1],
                      layout = "auto">
!array1 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3],
                      layout = "auto">
func.func @no_auto_layout(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CopyArrays' op does not allow input arrays with `auto` layout}}
  %0, %ctrl = ifrt.CopyArrays(%arg0) {donated=true}
    : (!array0) -> (!array1)
  return
}
