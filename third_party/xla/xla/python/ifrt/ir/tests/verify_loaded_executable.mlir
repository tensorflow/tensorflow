// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
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
