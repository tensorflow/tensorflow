// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_call(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @good_call_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>,
    %arg1: !ifrt.control)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) after %arg1 on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @good_call_with_io_aliases(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    {io_aliases=[array<i32: 0, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

#devices = #ifrt<devices[0,1]>
func.func @good_call_with_aliased_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       #devices>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices #devices
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   #devices>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   #devices>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----


func.func @call_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op must be in a FuncOp with attr `ifrt.function`}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @call_requires_valid_reference(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires '@missing_reference' to reference a valid `func.func`}}
  %0, %ctrl_0 = ifrt.Call @missing_reference(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

// -----

func.func @call_requires_same_input_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same input size. Input 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee() -> (tensor<4x4xi32>) {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_input_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<2x2xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x4xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_output_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same output size. Output 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) {
  return
}

// -----

func.func @call_requires_same_output_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Output #0 'tensor<4x4xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @call_requires_non_negative_devices_attr(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' Device list has negative logical id -1}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1,-1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_unique_devices_attr(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' Device list has duplicate logical id 0}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,0]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_input_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,2]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires all inputs placed on `devices` attr. The following input is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,2]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_output_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires all outputs placed on `devices` attr. The following output is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,2]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @io_aliases_should_be_pairs(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+2 {{'ifrt.Call' op attribute 'io_aliases' failed to satisfy constraint: Array of pairs of aliased input/output indices}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    {io_aliases=[array<i32: 0>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_valid_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    {io_aliases=[array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_only_alias_input_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 more than once}}
  %0, %1, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    {io_aliases=[array<i32: 0, 0>, array<i32: 0, 1>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                    [0,1]>,
        !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                    [0,1]>)
  return
}

func.func @callee(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32>, tensor<2x2xi32>) {
  return %arg0, %arg0 : tensor<2x2xi32>, tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_valid_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    {io_aliases=[array<i32: 0, 1>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_only_alias_output_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias output #0 more than once}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg0) on devices [0,1]
    {io_aliases=[array<i32: 0, 0>, array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>,
       !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_same_type(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different types: '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>' vs '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    {io_aliases=[array<i32: 0, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @good_call_local_view(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1,2,3] {ifrt.local_view}
    : (!ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    -> !ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @call_local_view_should_have_valid_shape(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<4x4xi32>' vs Callee 'tensor<8x8xi32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1,2,3] {ifrt.local_view}
    : (!ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    -> !ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>
  return
}

func.func @callee(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  return %arg0 : tensor<4x4xi32>
}