// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

ifrt.LoadedExecutable @good {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all inputs to be IfrtArrayType. Found 'tensor<2x2xi32>'}}
ifrt.LoadedExecutable @requires_array_input {devices=array<i64: 0, 1>}
    : (tensor<2x2xi32>) -> ()

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all outputs to be IfrtArrayType. Found 'tensor<2x2xi32>'}}
ifrt.LoadedExecutable @requires_array_output {devices=array<i64: 0, 1>}
    : () -> tensor<2x2xi32>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op has duplicate device id 0 in `devices` attr}}
ifrt.LoadedExecutable @requires_unique_devices_attr {devices=array<i64: 0, 0>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all inputs placed on `devices` attr. The following input is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0, 2]>'}}
ifrt.LoadedExecutable @requires_input_place_on_devices
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,2]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all outputs placed on `devices` attr. The following output is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0, 2]>'}}
ifrt.LoadedExecutable @requires_output_place_on_devices
    {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,2]>
