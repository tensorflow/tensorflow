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
// RUN: ifrt-opt %s -ifrt-lower-atom-program-metadata-to-xla='compile_options={{"compile_option_overrides": {"test_override": {"executable_build_options": {"use_shardy_partitioner": true}}}}}' -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// CHECK-LABEL: @arg_metadata_sdy_partitioned
module @arg_metadata_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: %arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG:    ifrt.memory_kind = "device"
  // CHECK-DAG:    mhlo.memory_kind = "device"
  // CHECK-SAME: }
  // CHECK: %arg1: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 2>
  // CHECK-SAME: }
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.memory_kind = "device"},
      %arg1: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<1x1 to [0] on 2>}) {
    return
  }
}

// -----

// CHECK-LABEL: @arg_unspecified_sharding_sdy_partitioned
module @arg_unspecified_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: %arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: }
  // CHECK: %arg1: tensor<2x2xi32> {ifrt.sharding = #ifrt.sharding_unspecified})
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>},
      %arg1: tensor<2x2xi32> {ifrt.sharding=#ifrt.sharding_unspecified}) {
    return
  }
}

// -----

// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @result_metadata_sdy_partitioned
module @result_metadata_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: -> (tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #sp
  // CHECK-DAG:    ifrt.memory_kind = "device"
  // CHECK-DAG:    mhlo.memory_kind = "device"
  // CHECK-SAME: }
  func.func @main()
      -> (tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.memory_kind = "device"}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @result_unspecified_sharding_sdy_partitioned
module @result_unspecified_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: -> (tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #sp
  // CHECK-SAME: }
  // CHECK: tensor<2x2xi32> {ifrt.sharding = #ifrt.sharding_unspecified})
  func.func @main()
      -> (tensor<2x2xi32> {
            ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>},
          tensor<2x2xi32> {ifrt.sharding=#ifrt.sharding_unspecified}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0, %0 : tensor<2x2xi32>, tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL: @arg_missing_sharding_sdy_partitioned
module @arg_missing_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: %arg0: tensor<2x2xi32>
  // CHECK-NOT: mhlo.sharding
  func.func @main(%arg0: tensor<2x2xi32>) {
    return
  }
}

// -----

// CHECK-LABEL: @result_missing_sharding_sdy_partitioned
module @result_missing_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: -> tensor<2x2xi32>
  // CHECK-NOT: mhlo.sharding
  func.func @main() -> (tensor<2x2xi32>) {
     %0 = mhlo.constant dense<1> : tensor<2x2xi32>
     return %0 : tensor<2x2xi32>
  }
}

// -----

// expected-error @+1 {{'builtin.module' op module `module_missing_devices_sdy_partitioned` must have `ifrt.num_devices` attribute}}
module @module_missing_devices_sdy_partitioned attributes {ifrt.compile_options_key = "test_override"} {
  func.func @main() -> (tensor<2x2xi32>
     {ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>,
       ifrt.devices=#ifrt<devices[1]>}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
