// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_call(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  %0, %ctrl_0 = ifrt.Call @good_call_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @good_call_callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @good_call_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>, %arg1: !ifrt.control) {
  %0, %ctrl_0 = ifrt.Call @good_call_with_control_dep_callee(%arg0) after %arg1
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @good_call_with_control_dep_callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @good_call_with_io_aliases(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 0, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @call_requires_valid_reference(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires '@missing_reference' to reference a valid function}}
  %0, %ctrl_0 = ifrt.Call @missing_reference(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

// -----

func.func @call_requires_same_input_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same input size. Input 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_input_size_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @call_requires_same_input_size_callee() -> (tensor<4x4xi32>) {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_input_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<2x2xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_input_shape_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @call_requires_same_input_shape_callee(%arg0: tensor<2x4xi32>)
   -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_output_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same output size. Output 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_output_size_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @call_requires_same_output_size_callee(%arg0: tensor<2x2xi32>) {
  return
}

// -----

func.func @call_requires_same_output_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Output #0 'tensor<4x4xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_output_shape_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @call_requires_same_output_shape_callee(%arg0: tensor<2x2xi32>)
    -> tensor<2x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @call_requires_unique_devices_attr(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op has duplicate device id 0 in `devices` attr}}
  %0, %ctrl_0 = ifrt.Call @call_requires_unique_devices_attr_callee(%arg0)
    {devices=array<i64: 0, 0>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @call_requires_unique_devices_attr_callee(%arg0: tensor<2x2xi32>)
    -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_input_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,2]>) {
  // expected-error@+1 {{'ifrt.Call' op requires all inputs placed on `devices` attr. The following input is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_input_place_on_devices_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,2]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

func.func @call_requires_input_place_on_devices_callee(%arg0: tensor<2x2xi32>)
    -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_output_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires all outputs placed on `devices` attr. The following output is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_output_place_on_devices_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,2]>
  return
}

func.func @call_requires_output_place_on_devices_callee(%arg0: tensor<2x2xi32>)
    -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @io_aliases_should_be_pairs(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op attribute 'io_aliases' failed to satisfy constraint: Array of pairs of aliased input/output indices}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_valid_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_only_alias_input_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 more than once}}
  %0, %1, %ctrl_0 = ifrt.Call @callee(%arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 0, 0>, array<i32: 0, 1>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>,
        !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
  return
}

func.func @callee(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32>, tensor<2x2xi32>) {
  return %arg0, %arg0 : tensor<2x2xi32>, tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_valid_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 0, 1>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_only_alias_output_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op can't alias output #0 more than once}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 0, 0>, array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>,
       !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_same_type(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different types: '!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0, 1]>' vs '!ifrt.array<tensor<2x2xi32>, 2x1 to [0] on 2, [0, 1]>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0)
    {devices=array<i64: 0, 1>, io_aliases=[array<i32: 0, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 2x1 to [0] on 2, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}
