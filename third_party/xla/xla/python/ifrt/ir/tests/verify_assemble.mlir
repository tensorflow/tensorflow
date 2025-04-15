// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_assemble(
    %arg0: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
    %arg1: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
    attributes {ifrt.function} {
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operandSegmentSizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
         !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
      -> !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @assemble_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
    %arg1: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [1]>) {
  // expected-error@+1 {{'ifrt.Assemble' op must be in a FuncOp with attr `ifrt.function`}}
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operandSegmentSizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
         !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
      -> !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
  return
}

// -----

func.func @assemble_requires_inputs_on_single_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>,
    %arg1: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [2]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Assemble' op requires every input to be a single device array. Actual: '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>'}}
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operandSegmentSizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>,
         !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [2]>)
      -> !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x3 to [0] on 3>, [0,1,2]>
  return
}

// -----

func.func @assemble_requires_same_device_list(
    %arg0: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
    %arg1: !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Assemble' op requires the same input/output device list. Input 0, 1 vs Output 1, 2}}
  %0 = "ifrt.Assemble"(%arg0, %arg1)
      {operandSegmentSizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
         !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
      -> !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [1,2]>
  return
}
