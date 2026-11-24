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
// RUN: ifrt-opt %s -ifrt-populate-atom-program-metadata -ifrt-duplicated-callee-elimination -symbol-dce -split-input-file -verify-diagnostics | FileCheck %s

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                     memory_kind = "pinned_host">
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @populate_arg_metadata
module @populate_arg_metadata {
  func.func @main(%arg0: !array) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0,1] : (!array) -> ()
    return
  }

  // CHECK: module @[[CALLEE]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-DAG: ifrt.memory_kind = "pinned_host"
  // CHECK-NOT: ifrt
  module @callee <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) {
      return
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @populate_sdy_partitioned_attr
module @populate_sdy_partitioned_attr {
  func.func @main(%arg0: !array) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0,1] : (!array) -> ()
    return
  }

  // CHECK: module @[[CALLEE]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-NOT: ifrt
  module @callee <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) {
      return
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @populate_compile_options_key_attr
module @populate_compile_options_key_attr {
  func.func @main(%arg0: !array) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0,1] {ifrt.compile_options_key = "foo"} : (!array) -> ()
    return
  }

  // CHECK: module @[[CALLEE]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.compile_options_key = "foo"
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-NOT: ifrt
  module @callee <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) {
      return
    }
  }
}

// -----

// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @populate_result_metadata
module @populate_result_metadata {
  func.func @main() attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE:.+]]::@main()
    %0, %1, %ctrl_0 = ifrt.Call @callee::@main() on devices [0,1]
        : () -> (!ifrt.array<tensor<2x2xi32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [1,0],
                             memory_kind = "pinned_host">,
                 !ifrt.array<tensor<2x2xi32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                             memory_kind = "device">)
    return
  }

  // CHECK: module @[[CALLEE]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main() -> (
  // CHECK-SAME: tensor<2x2xi32> {ifrt.devices = #ifrt<devices[1, 0]>, ifrt.memory_kind = "pinned_host", ifrt.sharding = #sp},
  // CHECK-SAME: tensor<2x2xi32> {ifrt.sharding = #sp})
  // CHECK-NOT: ifrt
  module @callee <sym_visibility = "private"> {
    func.func @main() -> (tensor<2x2xi32>, tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      return %0, %0 : tensor<2x2xi32>, tensor<2x2xi32>
    }
  }
}

// -----

// Verifies that a single module is populated with metadata if the input and
// output types are the same.
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @calls_outlined_to_single_module
module @calls_outlined_to_single_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // All calls share one metadata key, so `@add_one` is populated in place
    // with zero cloning and preserves its original symbol name.
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @add_one::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: %[[OUT_1:.+]], %[[CTRL_1:.+]] = ifrt.Call @add_one::@main(%[[OUT_0]])
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @add_one::@main(%[[OUT_1]]) after %[[CTRL_1]]
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%1) after %ctrl_1 on devices [0,1]
        : (!array) -> !array
    return %1 : !array
  }

  // CHECK: module @add_one <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-NOT: module @add_one_
  // CHECK-NOT: ifrt
  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

// Verifies that a single module is populated with metadata even if the
// devices are different.
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @calls_on_different_devices_outlined_to_single_module
module @calls_on_different_devices_outlined_to_single_module {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array0) -> !array0
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CopyArrays(%[[OUT_0]])
    %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array0) -> (!array1)
    // CHECK: %[[OUT_2:.+]], %[[CTRL_2:.+]] = ifrt.Call @[[CALLEE]]::@main(%[[OUT_1]])
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%1) on devices [2,3]
        : (!array1) -> !array1
    // CHECK: ifrt.Call @[[CALLEE]]::@main(%[[OUT_2]]) after %[[CTRL_2]]
    %3, %ctrl_3 = ifrt.Call @add_one::@main(%2) after %ctrl_2 on devices [2,3]
        : (!array1) -> !array1
    return %3 : !array1
  }

  // CHECK: module @[[CALLEE]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-NOT: ifrt
  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @call_twice_with_different_sharding
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0,1]>
module @call_twice_with_different_sharding {
  func.func @main(%arg0: !array) -> !array_unspecified
      attributes {ifrt.function} {
    // CHECK: %[[OUTPUT:.+]], %{{.+}} = ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[CALLEE_1:.+]]::@main(%[[OUTPUT]])
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array_unspecified
    return %1 : !array_unspecified
  }

  // CHECK: module @[[CALLEE_0]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp

  // CHECK: module @[[CALLEE_1]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_unspecified
  // CHECK-NOT: ifrt
  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @populate_io_alias_and_donation
module @populate_io_alias_and_donation {
  func.func @main(%arg0: !array, %arg1: !array) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0, %arg1)
    %0, %ctrl_0 = ifrt.Call @callee::@main(%arg0, %arg1) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>], donated_input_indices=[1]>
        : (!array, !array) -> !array
    // Verify that the module is cloned if io_aliases differ.
    // CHECK: ifrt.Call @[[CALLEE_1:.+]]::@main(%arg0, %arg1)
    %1, %ctrl_1 = ifrt.Call @callee::@main(%arg0, %arg1) on devices [0,1]
        : (!array, !array) -> !array
    return
  }

  // CHECK: module @[[CALLEE_0]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-DAG:     ifrt.sharding = #sp
  // CHECK-DAG:     tf.aliasing_output = 0 : i32
  // CHECK-SAME: }
  // CHECK: %arg1: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-DAG:     ifrt.sharding = #sp
  // CHECK-DAG:     jax.buffer_donor = true
  // CHECK-SAME: }

  // CHECK: module @[[CALLEE_1]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-DAG:     ifrt.sharding = #sp
  // CHECK-SAME: }
  module @callee <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
         -> tensor<2x2xi32> {
      return %arg0: tensor<2x2xi32>
    }
  }
}

// -----

!shared_array = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @output_of_call_donated
module @output_of_call_donated {
  func.func @main(%arg0: !shared_array) -> !shared_array
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0) on devices [0, 1] :
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!shared_array) -> !shared_array
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.Call @[[CALLEE_1:.+]]::@main(%[[OUT_0]]) on devices [0, 1] <io_aliases = [array<i32: 0, 0>]> :
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!shared_array) -> !shared_array
    return %1 : !shared_array
  }

  // CHECK: module @[[CALLEE_0]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>

  // CHECK: module @[[CALLEE_1]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>
  // CHECK-SAME: tf.aliasing_output = 0 : i32
  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2,3],
                      memory_kind = "device">
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [3,2],
                      memory_kind = "device">
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @populate_device_attr_permutation
module @populate_device_attr_permutation {
  func.func @main(%arg0: !array0, %arg1: !array1) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0, %arg1)
    %ctrl_0 = ifrt.Call @callee::@main(%arg0, %arg1) on devices [2,3] : (!array0, !array1) -> ()
    // CHECK: ifrt.Call @[[CALLEE_1:.+]]::@main(%arg0, %arg1)
    %ctrl_1 = ifrt.Call @callee::@main(%arg0, %arg1) on devices [3,2] : (!array0, !array1) -> ()
    return
  }

  // CHECK: module @[[CALLEE_0]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32> {ifrt.sharding = #sp},
  // CHECK-SAME: %arg1: tensor<2x2xi32> {ifrt.devices = #ifrt<devices[1, 0]>, ifrt.sharding = #sp})

  // CHECK: module @[[CALLEE_1]] <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32> {ifrt.devices = #ifrt<devices[1, 0]>, ifrt.sharding = #sp},
  // CHECK-SAME: %arg1: tensor<2x2xi32> {ifrt.sharding = #sp})
  // CHECK-NOT: ifrt
  module @callee <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) {
      return
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @recursive_outlined_callee_isolates_internal_uses
module @recursive_outlined_callee_isolates_internal_uses {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.Call @rec_callee::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @rec_callee::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  // CHECK: module @rec_callee <sym_visibility = "private">
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-SAME: }
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK: call @[[HELPER:main_.+]](%arg0)
  // CHECK: func.func private @[[HELPER]](%arg0: tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK-NOT: ifrt.sharding
  // CHECK: call @[[HELPER]](%arg0)
  // CHECK-NOT: module @rec_callee_
  module @rec_callee <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = func.call @main(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
      return %0 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @interleaved_calls_claim_earliest_variant_in_place
module @interleaved_calls_claim_earliest_variant_in_place {
  func.func @main(%arg0: !array) -> (!array, !array_unspecified, !array)
      attributes {ifrt.function} {
    // Call 0 and Call 2 share the earliest variant key, so `@add_one` is
    // claimed in place while Call 1 clones once into `@add_one_0`.
    // CHECK: %[[OUT0:.+]], %{{.+}} = ifrt.Call @add_one::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: %[[OUT1:.+]], %{{.+}} = ifrt.Call @[[CLONED:add_one_.+]]::@main(%[[OUT0]])
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array_unspecified
    // CHECK: %[[OUT2:.+]], %{{.+}} = ifrt.Call @add_one::@main(%[[OUT0]])
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array
    return %0, %1, %2 : !array, !array_unspecified, !array
  }

  // CHECK: module @add_one <sym_visibility = "private">
  // CHECK-SAME: ifrt.num_devices = 2
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp

  // CHECK: module @[[CLONED]] <sym_visibility = "private">
  // CHECK-SAME: ifrt.num_devices = 2
  // CHECK: func.func @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #sp
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_unspecified
  // CHECK-NOT: module @add_one_
  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
