// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_call(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  %0, %ctrl_0 = ifrt.Call @good_call_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @good_call_callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @good_call_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>, %arg1: !ifrt.control) {
  %0, %ctrl_0 = ifrt.Call @good_call_with_control_dep_callee(%arg0) after %arg1
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @good_call_with_control_dep_callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_valid_reference(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires '@missing_reference' to reference a valid function}}
  %0, %ctrl_0 = ifrt.Call @missing_reference(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

// -----

func.func @call_requires_same_input_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same input size. Input 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_input_size_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @call_requires_same_input_size_callee() -> (tensor<4x4xi32>) {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_input_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<2x2xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_input_shape_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @call_requires_same_input_shape_callee(%arg0: tensor<2x4xi32>)
   -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_output_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same output size. Output 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_output_size_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @call_requires_same_output_size_callee(%arg0: tensor<2x2xi32>) {
  return
}

// -----

func.func @call_requires_same_output_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Output #0 'tensor<4x4xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_same_output_shape_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @call_requires_same_output_shape_callee(%arg0: tensor<2x2xi32>)
    -> tensor<2x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @call_requires_unique_devices_attr(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op has duplicated device id 0 in `devices` attr}}
  %0, %ctrl_0 = ifrt.Call @call_requires_unique_devices_attr_callee(%arg0)
    {devices=array<i64: 0, 0>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @call_requires_unique_devices_attr_callee(%arg0: tensor<2x2xi32>)
    -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_input_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,2]>) {
  // expected-error@+1 {{'ifrt.Call' op requires all inputs placed on `devices` attr. The following input is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<2x2xi32>, "shard0", [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_input_place_on_devices_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,2]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,1]>
  return
}

func.func @call_requires_input_place_on_devices_callee(%arg0: tensor<2x2xi32>)
    -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
// -----

func.func @call_requires_output_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op requires all outputs placed on `devices` attr. The following output is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<4x4xi32>, "shard1", [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @call_requires_output_place_on_devices_callee(%arg0)
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, "shard0", [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, "shard1", [0,2]>
  return
}

func.func @call_requires_output_place_on_devices_callee(%arg0: tensor<2x2xi32>)
    -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
