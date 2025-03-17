// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_reshard(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0, %ctrl = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 4>,
                     [0,1,2,3]>
  return
}

func.func @good_reshard_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>,
    %arg1: !ifrt.control)
    attributes {ifrt.function} {
  %0, %ctrl = ifrt.Reshard(%arg0) after %arg1
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 4>,
                     [0,1,2,3]>
  return
}

!array0 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 2>, [0,1]>
!array2 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 2>, [2,3]>
func.func @good_reshard_with_two_arrays(%arg0: !array0) -> (!array1, !array2)
    attributes {ifrt.function} {
  %0, %1, %ctrl_1 = ifrt.Reshard(%arg0, %arg0)
      : (!array0, !array0) -> (!array1, !array2)
  return %0, %1 : !array1, !array2
}

// -----

func.func @reshard_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>) {
  // expected-error@+1 {{'ifrt.Reshard' op must be in a FuncOp with attr `ifrt.function`}}
  %0, %ctrl = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 4>,
                     [0,1,2,3]>
  return
}

// -----

func.func @reshard_requires_same_global_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Reshard' op requires the same global shape. input #0 'tensor<2x2xi32>' vs output #0 'tensor<2x1xi32>'}}
  %0, %ctrl = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x1xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [2,3]>
  return
}

// -----

func.func @reshard_requires_non_negative_axis_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+5 {{Out of range axis -1 to the mesh of [-1] on 2}}
  // expected-error@+4 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0, %ctrl = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x1xi32>, #ifrt.sharding_param<1x2 to [-1] on 2>,
                     [2,3]>
  return
}

// -----

func.func @reshard_requires_valid_axis_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+6 {{Out of range axis 1234567890 to the mesh of [1234567890] on 2}}
  // expected-error@+5 {{failed to parse Ifrt_ArrayType parameter 'sharding_attr'}}
  %0, %ctrl = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x1xi32>,
                     #ifrt.sharding_param<1x2 to [1234567890] on 2>, [2,3]>
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_at_least_one_input() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Reshard' op requires at least one input array}}
  %ctrl = ifrt.Reshard() : () -> ()
  return
}

// -----

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_same_num_inputs_and_outputs(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Reshard' op requires the same number of input and output arrays}}
  %0, %1, %ctrl = ifrt.Reshard(%arg0) : (!array) -> (!array, !array)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xf32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
func.func @requires_resharded_array_to_have_same_dtype(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Reshard' op requires the same global shape. input #0 'tensor<2x4xi32>' vs output #0 'tensor<2x4xf32>'}}
  %0, %ctrl = ifrt.Reshard(%arg0) : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                      [0,1], layout = "auto">
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 4>,
                      [0,1,2,3], layout = "auto">
func.func @no_auto_layout(%arg0: !array0) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Reshard' op does not allow input or output arrays with `auto` layout}}
  %0, %ctrl = ifrt.Reshard(%arg0) : (!array0) -> !array1
  return
}

