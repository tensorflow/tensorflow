// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

ifrt.LoadedExecutable @good on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

#devices = #ifrt<devices[0,1]>
ifrt.LoadedExecutable @good_with_aliased_devices on devices #devices
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   #devices>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   #devices>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all inputs to be IfrtArrayType. Found 'tensor<2x2xi32>'}}
ifrt.LoadedExecutable @requires_array_input on devices [0,1]
    : (tensor<2x2xi32>) -> ()

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all outputs to be IfrtArrayType. Found 'tensor<2x2xi32>'}}
ifrt.LoadedExecutable @requires_array_output on devices [0,1]
    : () -> tensor<2x2xi32>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' Device list has duplicate logical id 0}}
ifrt.LoadedExecutable @requires_unique_devices_attr on devices [0,0]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all inputs placed on `devices` attr. The following input is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 2]>'}}
ifrt.LoadedExecutable @requires_input_place_on_devices on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,2]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

// -----

// expected-error@+1 {{'ifrt.LoadedExecutable' op requires all outputs placed on `devices` attr. The following output is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 2]>'}}
ifrt.LoadedExecutable @requires_output_place_on_devices on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,2]>
