// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_reshard(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    attributes {ifrt.function} {
  %0 = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 4, [0,1,2,3]>
  return
}

func.func @good_reshard_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>,
    %arg1: !ifrt.control)
    attributes {ifrt.function} {
  %0 = ifrt.Reshard(%arg0) after %arg1
      : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 4, [0,1,2,3]>
  return
}

// -----

func.func @reshard_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Reshard' op must be in a FuncOp with attr `ifrt.function`}}
  %0 = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 4, [0,1,2,3]>
  return
}

// -----

func.func @reshard_requires_same_global_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Reshard' op requires the same global shape. Input 'tensor<2x2xi32>' vs Output 'tensor<2x1xi32>'}}
  %0 = ifrt.Reshard(%arg0)
      : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
      -> !ifrt.array<tensor<2x1xi32>, 1x1 to [0] on 2, [2,3]>
  return
}
