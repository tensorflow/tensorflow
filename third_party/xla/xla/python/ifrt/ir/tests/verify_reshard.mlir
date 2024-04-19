// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_reshard(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0 = ifrt.Reshard(%arg0)
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
  %0 = ifrt.Reshard(%arg0) after %arg1
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 4>,
                     [0,1,2,3]>
  return
}

// -----

func.func @reshard_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>) {
  // expected-error@+1 {{'ifrt.Reshard' op must be in a FuncOp with attr `ifrt.function`}}
  %0 = ifrt.Reshard(%arg0)
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
  // expected-error@+1 {{'ifrt.Reshard' op requires the same global shape. Input 'tensor<2x2xi32>' vs Output 'tensor<2x1xi32>'}}
  %0 = ifrt.Reshard(%arg0)
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
  %0 = ifrt.Reshard(%arg0)
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
  %0 = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x1xi32>,
                     #ifrt.sharding_param<1x2 to [1234567890] on 2>, [2,3]>
  return
}
