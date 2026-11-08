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
// RUN: ifrt-opt %s -ifrt-call-dedup -split-input-file -verify-diagnostics | FileCheck %s

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_operands
module @duplicate_operands {
  func.func @main(%arg0: !array, %arg1: !array, %arg2: !array)
      -> (!array, !array, !array) attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:3, %{{.*}} = ifrt.Call @f(%arg0, %arg1, %arg2, %arg0, %arg2, %arg2) on devices [0, 1]
    %0:3, %ctrl_0 = ifrt.Call @f(%arg0, %arg1, %arg2, %arg0, %arg2, %arg2)
        on devices [0, 1]
        : (!array, !array, !array, !array, !array, !array)
        -> (!array, !array, !array)
    // CHECK: return %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2
    return %0#0, %0#1, %0#2 : !array, !array, !array
  }

  // CHECK-LABEL: func.func private @f(
  func.func private @f(
      %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>,
      %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
    // %arg0, %arg1, %arg2 are the unique operands. They should be used instead
    // of %arg3, %arg4, %arg5.
    // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %arg2, %arg0 : tensor<4x8xf32>
    // CHECK-NEXT: %[[ADD2:.*]] = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    // CHECK-NEXT: return %[[ADD0]], %[[ADD1]], %[[ADD2]]
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.add %arg4, %arg5 : tensor<4x8xf32>
    return %0, %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_results
module @duplicate_results {
  func.func @main(%arg0: !array, %arg1: !array)
      -> (!array, !array, !array, !array) attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:4, %{{.*}} = ifrt.Call @f(%arg0, %arg1) on devices [0, 1]
    %0:4, %ctrl_0 = ifrt.Call @f(%arg0, %arg1) on devices [0, 1]
        : (!array, !array) -> (!array, !array, !array, !array)
    // Only use the first result since all the others are duplicates.
    // CHECK: return %[[OUT]]#0, %[[OUT]]#0, %[[OUT]]#0, %[[OUT]]#0
    return %0#0, %0#1, %0#2, %0#3 : !array, !array, !array, !array
  }

  func.func private @f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0, %0, %0, %0
        : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array_dist_x = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_dist_y = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_results_but_with_different_types
module @duplicate_results_but_with_different_types {
  func.func @main(%arg0: !array_dist_x) -> (!array_dist_x, !array_dist_y)
      attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:2, %{{.*}} = ifrt.Call @f(%arg0) on devices [0, 1]
    %0:2, %ctrl_0 = ifrt.Call @f(%arg0) on devices [0, 1]
        : (!array_dist_x) -> (!array_dist_x, !array_dist_y)
    // CHECK: return %[[OUT]]#0, %[[OUT]]#1
    return %0#0, %0#1 : !array_dist_x, !array_dist_y
  }

  // CHECK-LABEL: func.func private @f(
  // CHECK-NEXT: return %arg0, %arg0
  func.func private @f(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    return %arg0, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_results_with_different_res_attrs
module @duplicate_results_with_different_res_attrs {
  func.func @main(%arg0: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:2, %{{.*}} = ifrt.Call @f(%arg0) on devices [0, 1]
    %0:2, %ctrl_0 = ifrt.Call @f(%arg0) on devices [0, 1]
        : (!array) -> (!array, !array)
    // CHECK: return %[[OUT]]#0, %[[OUT]]#1
    return %0#0, %0#1 : !array, !array
  }

  func.func private @f(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32> {mhlo.layout_mode = "auto"},
          tensor<4x8xf32> {mhlo.layout_mode = "default"}) {
    return %arg0, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_operands_with_different_arg_attrs
module @duplicate_operands_with_different_arg_attrs {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @f(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    return %0 : !array
  }

  // CHECK-LABEL: func.func private @f(
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD]]
  func.func private @f(
      %arg0: tensor<4x8xf32> {mhlo.layout_mode = "auto"},
      %arg1: tensor<4x8xf32> {mhlo.layout_mode = "default"})
      -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_operands_becoming_duplicate_results_and_cascading
module @duplicate_operands_becoming_duplicate_results_and_cascading {
  func.func @main(%arg0: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[PROD:.*]]:2, %{{.*}} = ifrt.Call @producer(%arg0, %arg0) on devices [0, 1]
    %0:2, %ctrl_0 = ifrt.Call @producer(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> (!array, !array)
    // CHECK: %[[CONS:.*]], %{{.*}} = ifrt.Call @consumer(%[[PROD]]#0, %[[PROD]]#0) on devices [0, 1]
    %1, %ctrl_1 = ifrt.Call @consumer(%0#0, %0#1) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: return %[[PROD]]#0, %[[CONS]]
    return %0#1, %1 : !array, !array
  }

  // CHECK-LABEL: func.func private @producer(
  // CHECK-NEXT: return %arg0, %arg0
  func.func private @producer(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    return %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @consumer(
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD]]
  func.func private @consumer(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @shared_callee_cloned_on_operand_dedup
module @shared_callee_cloned_on_operand_dedup {
  func.func @main(%arg0: !array, %arg1: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[OUT0:.*]], %{{.*}} = ifrt.Call @[[CLONED:shared_f_.+]](%arg0, %arg0) on devices [0, 1]
    %0, %ctrl_0 = ifrt.Call @shared_f(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: %[[OUT1:.*]], %{{.*}} = ifrt.Call @shared_f(%arg0, %arg1) on devices [0, 1]
    %1, %ctrl_1 = ifrt.Call @shared_f(%arg0, %arg1) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: return %[[OUT0]], %[[OUT1]]
    return %0, %1 : !array, !array
  }

  // CHECK-LABEL: func.func private @shared_f(
  // CHECK-NEXT: %[[ORIG_ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ORIG_ADD]]
  func.func private @shared_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @shared_f_
  // CHECK-NEXT: %[[CLONED_ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[CLONED_ADD]]
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @aliased_input_and_output_not_deduplicated
module @aliased_input_and_output_not_deduplicated {
  func.func @main(%arg0: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:2, %{{.*}} = ifrt.Call @f(%arg0, %arg0) on devices [0, 1] <io_aliases = [array<i32: 1, 1>]>
    %0:2, %ctrl_0 = ifrt.Call @f(%arg0, %arg0) on devices [0, 1]
        <io_aliases = [array<i32: 1, 1>]>
        : (!array, !array) -> (!array, !array)
    // CHECK: return %[[OUT]]#0, %[[OUT]]#1
    return %0#0, %0#1 : !array, !array
  }

  // CHECK-LABEL: func.func private @f(
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD]], %[[ADD]]
  func.func private @f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0, %0 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @duplicate_output_donated_to_call_not_deduplicated
module @duplicate_output_donated_to_call_not_deduplicated {
  func.func @main(%arg0: !array) -> (!array, !array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[PROD:.*]]:3, %{{.*}} = ifrt.Call @producer(%arg0) on devices [0, 1]
    %0:3, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1]
        : (!array) -> (!array, !array, !array)
    // CHECK: %[[CONS:.*]], %{{.*}} = ifrt.Call @consumer(%[[PROD]]#0) on devices [0, 1] <donated_input_indices = [0]>
    %1, %ctrl_1 = ifrt.Call @consumer(%0#0) on devices [0, 1]
        <donated_input_indices = [0]>
        : (!array) -> !array
    // %0#0 is donated to @consumer so %0#1 must not merge into %0#0, while
    // %0#1 and %0#2 still deduplicate to %0#1.
    // CHECK: return %[[CONS]], %[[PROD]]#1, %[[PROD]]#1
    return %1, %0#1, %0#2 : !array, !array, !array
  }

  func.func private @producer(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
    return %arg0, %arg0, %arg0
        : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }

  func.func private @consumer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @later_duplicate_output_aliased_to_loaded_executable_not_deduplicated
module @later_duplicate_output_aliased_to_loaded_executable_not_deduplicated {
  func.func @main(%arg0: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[PROD:.*]]:2, %{{.*}} = ifrt.Call @producer(%arg0) on devices [0, 1]
    %0:2, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1]
        : (!array) -> (!array, !array)
    // CHECK: %[[EXEC:.*]], %{{.*}} = ifrt.CallLoadedExecutable @loaded_exec(%[[PROD]]#1) <io_aliases = [array<i32: 0, 0>]>
    %1, %ctrl_1 = ifrt.CallLoadedExecutable @loaded_exec(%0#1)
        <io_aliases = [array<i32: 0, 0>]>
        : (!array) -> !array
    // CHECK: return %[[PROD]]#0, %[[EXEC]]
    return %0#0, %1 : !array, !array
  }

  func.func private @producer(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    return %arg0, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>
  }

  ifrt.LoadedExecutable @loaded_exec on devices [0, 1] : (!array) -> !array
}

// -----

!array0 = !ifrt.array<tensor<4x8xf32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<4x8xf32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2, 3]>
!array_bitcast = !ifrt.array<tensor<8x4xf32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<2x8xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<2x8xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>

// CHECK-LABEL: module @duplicate_outputs_donated_or_aliased_to_transfer_and_metadata_ops_not_deduplicated
module @duplicate_outputs_donated_or_aliased_to_transfer_and_metadata_ops_not_deduplicated {
  func.func @main(%arg0: !array0, %arg1: !shard1)
      -> (!array0, !array1, !array0, !array1, !array_bitcast, !array0, !array0,
          !shard0, !shard1, !array1, !array1, !shard0, !array0)
      attributes {ifrt.function} {
    // CHECK: %[[PROD:.*]]:10, %{{.*}} = ifrt.Call @producer(%arg0) on devices [0, 1]
    %0:10, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1]
        : (!array0) -> (!array0, !array0, !array0, !array0, !array0,
                        !array0, !array0, !array0, !array0, !array0)
    // CHECK: %[[COPY_DONATED:.*]], %{{.*}} = ifrt.CopyArrays(%[[PROD]]#1) <donated = true>
    %c1, %ctrl_1 = ifrt.CopyArrays(%0#1) <donated = true>
        : (!array0) -> !array1
    // CHECK: %[[COPY_REUSE:.*]], %{{.*}} = ifrt.CopyArrays(%[[PROD]]#2) <reuse = true>
    %c2, %ctrl_2 = ifrt.CopyArrays(%0#2) <reuse = true>
        : (!array0) -> !array0
    // CHECK: %[[RESHARD_DONATED:.*]], %{{.*}} = ifrt.Reshard(%[[PROD]]#3) <donated = true>
    %r3, %ctrl_3 = ifrt.Reshard(%0#3) <donated = true>
        : (!array0) -> !array1
    // CHECK: %[[BITCAST_DONATED:.*]], %{{.*}} = ifrt.BitcastArrays(%[[PROD]]#4) <donated = true>
    %b4, %ctrl_4 = ifrt.BitcastArrays(%0#4) <donated = true>
        : (!array0) -> !array_bitcast
    // CHECK: %[[REMAP_DONATED:.*]], %{{.*}} = ifrt.RemapArrays(%[[PROD]]#5)
    %m5, %ctrl_5 = ifrt.RemapArrays(%0#5)
        mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:2:1] to [0:2:1]>]>]
        <donated = true> : (!array0) -> !array0
    // Non-donating RemapArrays still aliases its input buffer.
    // CHECK: %[[REMAP_ALIASED:.*]], %{{.*}} = ifrt.RemapArrays(%[[PROD]]#6)
    %m6, %ctrl_6 = ifrt.RemapArrays(%0#6)
        mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:2:1] to [0:2:1]>]>]
        : (!array0) -> !array0
    // Disassemble creates shard views aliasing its input buffer.
    // CHECK: %[[DIS:.*]]:2, %{{.*}} = ifrt.Disassemble(%[[PROD]]#7)
    %d0, %d1, %ctrl_7 = ifrt.Disassemble(%0#7)
        : (!array0) -> (!shard0, !shard1)
    // Pure CopyArrays (donated=false, reuse=false) and pure Reshard
    // (donated=false) allocate fresh buffers and still deduplicate to %0#0.
    // CHECK: %[[COPY_PURE:.*]], %{{.*}} = ifrt.CopyArrays(%[[PROD]]#0) :
    %c8, %ctrl_8 = ifrt.CopyArrays(%0#8) : (!array0) -> !array1
    // CHECK: %[[RESHARD_PURE:.*]], %{{.*}} = ifrt.Reshard(%[[PROD]]#0) :
    %r9, %ctrl_9 = ifrt.Reshard(%0#9) : (!array0) -> !array1

    // Assemble aliases its single-device input buffers.
    // CHECK: %[[SHARD_PROD:.*]]:2, %{{.*}} = ifrt.Call @shard_producer(%[[DIS]]#0) on devices [0]
    %s:2, %ctrl_10 = ifrt.Call @shard_producer(%d0) on devices [0]
        : (!shard0) -> (!shard0, !shard0)
    // CHECK: %[[ASM:.*]], %{{.*}} = ifrt.Assemble(%[[SHARD_PROD]]#1, %arg1)
    %a, %ctrl_11 = ifrt.Assemble(%s#1, %arg1)
        : (!shard0, !shard1) -> !array0

    // CHECK: return %[[PROD]]#0, %[[COPY_DONATED]], %[[COPY_REUSE]], %[[RESHARD_DONATED]], %[[BITCAST_DONATED]], %[[REMAP_DONATED]], %[[REMAP_ALIASED]], %[[DIS]]#0, %[[DIS]]#1, %[[COPY_PURE]], %[[RESHARD_PURE]], %[[SHARD_PROD]]#0, %[[ASM]]
    return %0#0, %c1, %c2, %r3, %b4, %m5, %m6, %d0, %d1, %c8, %r9, %s#0, %a
        : !array0, !array1, !array0, !array1, !array_bitcast, !array0, !array0,
          !shard0, !shard1, !array1, !array1, !shard0, !array0
  }

  func.func private @producer(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
          tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
          tensor<4x8xf32>, tensor<4x8xf32>) {
    return %arg0, %arg0, %arg0, %arg0, %arg0,
           %arg0, %arg0, %arg0, %arg0, %arg0
        : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
          tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
          tensor<4x8xf32>, tensor<4x8xf32>
  }

  func.func private @shard_producer(%arg0: tensor<2x8xf32>)
      -> (tensor<2x8xf32>, tensor<2x8xf32>) {
    return %arg0, %arg0 : tensor<2x8xf32>, tensor<2x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @callee_also_used_by_func_call_in_atom_helper_is_cloned
module @callee_also_used_by_func_call_in_atom_helper_is_cloned {
  func.func @main(%arg0: !array, %arg1: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[OUT0:.*]], %{{.*}} = ifrt.Call @[[CLONED:shared_helper_.+]](%arg0, %arg0) on devices [0, 1]
    %0, %ctrl_0 = ifrt.Call @shared_helper(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: %[[OUT1:.*]], %{{.*}} = ifrt.Call @other_atom(%arg0, %arg1) on devices [0, 1]
    %1, %ctrl_1 = ifrt.Call @other_atom(%arg0, %arg1) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: return %[[OUT0]], %[[OUT1]]
    return %0, %1 : !array, !array
  }

  // CHECK-LABEL: func.func private @other_atom(
  // CHECK-NEXT: %[[CALL:.*]] = call @shared_helper(%arg0, %arg1)
  // CHECK-NEXT: return %[[CALL]]
  func.func private @other_atom(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = func.call @shared_helper(%arg0, %arg1)
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @shared_helper(
  // CHECK-NEXT: %[[ORIG_ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ORIG_ADD]]
  func.func private @shared_helper(
      %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @shared_helper_
  // CHECK-NEXT: %[[CLONED_ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[CLONED_ADD]]
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @public_callee_and_public_outlined_module_cloned_as_private
module @public_callee_and_public_outlined_module_cloned_as_private {
  func.func @main(%arg0: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[OUT0:.*]], %{{.*}} = ifrt.Call @[[CLONED_FN:public_f_.+]](%arg0, %arg0) on devices [0, 1]
    %0, %ctrl_0 = ifrt.Call @public_f(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: %[[OUT1:.*]], %{{.*}} = ifrt.Call @[[CLONED_MOD:public_mod_.+]]::@main(%arg0, %arg0) on devices [0, 1]
    %1, %ctrl_1 = ifrt.Call @public_mod::@main(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: return %[[OUT0]], %[[OUT1]]
    return %0, %1 : !array, !array
  }

  // CHECK-LABEL: func.func @public_f(
  // CHECK-NEXT: %[[PUB_ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[PUB_ADD]]
  func.func @public_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @public_f_
  // CHECK-NEXT: %[[PRIV_ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[PRIV_ADD]]

  // CHECK-LABEL: module @public_mod {
  // CHECK: func.func @main(
  // CHECK-NEXT: %[[MOD_PUB_ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[MOD_PUB_ADD]]
  module @public_mod {
    func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
        -> tensor<4x8xf32> {
      %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
      return %0 : tensor<4x8xf32>
    }
  }

  // CHECK-LABEL: module @public_mod_
  // CHECK-SAME: <sym_visibility = "private">
  // CHECK: func.func @main(
  // CHECK-NEXT: %[[MOD_PRIV_ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[MOD_PRIV_ADD]]
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @outlined_callee_used_by_internal_func_call_is_specialized
module @outlined_callee_used_by_internal_func_call_is_specialized {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @[[CLONED_MOD:outlined_mod_.+]]::@main(%arg0, %arg0) on devices [0, 1]
    %0, %ctrl_0 = ifrt.Call @outlined_mod::@main(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: return %[[OUT]]
    return %0 : !array
  }

  // CHECK-LABEL: module @outlined_mod <sym_visibility = "private">
  // CHECK: func.func @main(
  // CHECK-NEXT: %[[ORIG_ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ORIG_ADD]]
  module @outlined_mod <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
        -> tensor<4x8xf32> {
      %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
      return %0 : tensor<4x8xf32>
    }

    func.func private @helper(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
        -> tensor<4x8xf32> {
      %0 = func.call @main(%arg0, %arg1)
          : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
      return %0 : tensor<4x8xf32>
    }
  }

  // CHECK-LABEL: module @outlined_mod_
  // CHECK-SAME: <sym_visibility = "private">
  // CHECK: func.func @main(
  // CHECK-NEXT: %[[SPEC_ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[SPEC_ADD]]
  // CHECK: func.func private @[[UNMUTATED_HELPER:main_.+]](
  // CHECK-NEXT: %[[INNER_ORIG_ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[INNER_ORIG_ADD]]
  // CHECK: func.func private @helper(
  // CHECK-NEXT: %[[HELPER_CALL:.*]] = call @[[UNMUTATED_HELPER]](%arg0, %arg1)
  // CHECK-NEXT: return %[[HELPER_CALL]]
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @shared_callee_with_identical_duplicate_pattern_cloned_once
module @shared_callee_with_identical_duplicate_pattern_cloned_once {
  func.func @main(%arg0: !array, %arg1: !array)
      -> (!array, !array, !array, !array) attributes {ifrt.function} {
    // CHECK: %[[S0:.*]]:2, %{{.*}} = ifrt.Call @[[CLONED:stage_fn_.+]](%arg0, %arg0) on devices [0, 1]
    %0:2, %ctrl_0 = ifrt.Call @stage_fn(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> (!array, !array)
    // Producer-to-consumer cascading rewrites %0#1 -> %0#0, and the second and
    // third stages reuse the same @stage_fn_ clone rather than cloning again.
    // CHECK: %[[S1:.*]]:2, %{{.*}} = ifrt.Call @[[CLONED]](%[[S0]]#0, %[[S0]]#0) on devices [0, 1]
    %1:2, %ctrl_1 = ifrt.Call @stage_fn(%0#0, %0#1) on devices [0, 1]
        : (!array, !array) -> (!array, !array)
    // CHECK: %[[S2:.*]]:2, %{{.*}} = ifrt.Call @[[CLONED]](%[[S1]]#0, %[[S1]]#0) on devices [0, 1]
    %2:2, %ctrl_2 = ifrt.Call @stage_fn(%1#0, %1#1) on devices [0, 1]
        : (!array, !array) -> (!array, !array)
    // CHECK: %[[S3:.*]]:2, %{{.*}} = ifrt.Call @stage_fn(%arg0, %arg1) on devices [0, 1]
    %3:2, %ctrl_3 = ifrt.Call @stage_fn(%arg0, %arg1) on devices [0, 1]
        : (!array, !array) -> (!array, !array)
    // CHECK: return %[[S0]]#0, %[[S1]]#0, %[[S2]]#0, %[[S3]]#1
    return %0#1, %1#1, %2#1, %3#1 : !array, !array, !array, !array
  }

  // CHECK-LABEL: func.func private @stage_fn(
  // CHECK-NEXT: return %arg0, %arg1
  func.func private @stage_fn(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    return %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @stage_fn_
  // CHECK-NEXT: return %arg0, %arg0
  // CHECK-NOT: func.func private @stage_fn_
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @donated_duplicate_input_not_deduplicated
module @donated_duplicate_input_not_deduplicated {
  func.func @main(%arg0: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:2, %{{.*}} = ifrt.Call @f(%arg0, %arg0, %arg0) on devices [0, 1] <donated_input_indices = [1]>
    %0:2, %ctrl_0 = ifrt.Call @f(%arg0, %arg0, %arg0) on devices [0, 1]
        <donated_input_indices = [1]>
        : (!array, !array, !array) -> (!array, !array)
    // CHECK: return %[[OUT]]#0, %[[OUT]]#1
    return %0#0, %0#1 : !array, !array
  }

  // Input #1 is donated so %arg1 must stay distinct, while non-donated
  // duplicate %arg2 merges into %arg0.
  // CHECK-LABEL: func.func private @f(
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD0]], %[[ADD1]]
  func.func private @f(
      %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    %1 = stablehlo.add %arg0, %arg2 : tensor<4x8xf32>
    return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @call_op_arg_and_res_attrs_and_empty_dict_normalization
module @call_op_arg_and_res_attrs_and_empty_dict_normalization {
  func.func @main(%arg0: !array) -> (!array, !array, !array, !array)
      attributes {ifrt.function} {
    // CallOp arg_attrs:
    //   #0: {} (normalizes to empty)
    //   #1: {mhlo.layout_mode = "auto"}
    //   #2: {mhlo.layout_mode = "auto"} (merges into #1)
    //   #3: {mhlo.layout_mode = "default"} (distinct from #0 and #1)
    //   #4: omitted / out-of-bounds (normalizes to empty, merges into #0)
    // CallOp res_attrs:
    //   #0: {} (normalizes to empty)
    //   #1: {mhlo.layout_mode = "auto"}
    //   #2: {mhlo.layout_mode = "auto"} (merges into #1)
    //   #3: omitted / out-of-bounds (normalizes to empty, merges into #0)
    // CHECK: %[[OUT:.*]]:4, %{{.*}} = ifrt.Call @f(%arg0, %arg0, %arg0, %arg0, %arg0) on devices [0, 1]
    %0:4, %ctrl_0 = ifrt.Call @f(%arg0, %arg0, %arg0, %arg0, %arg0)
        on devices [0, 1]
        <arg_attrs = [{}, {mhlo.layout_mode = "auto"},
                      {mhlo.layout_mode = "auto"},
                      {mhlo.layout_mode = "default"}],
         res_attrs = [{}, {mhlo.layout_mode = "auto"},
                      {mhlo.layout_mode = "auto"}]>
        : (!array, !array, !array, !array, !array)
        -> (!array, !array, !array, !array)
    // CHECK: return %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#1, %[[OUT]]#0
    return %0#0, %0#1, %0#2, %0#3 : !array, !array, !array, !array
  }

  // CHECK-LABEL: func.func private @f(
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %arg1, %arg3 : tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD2:.*]] = stablehlo.add %[[ADD0]], %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD3:.*]] = stablehlo.add %[[ADD1]], %[[ADD2]] : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD3]], %[[ADD3]], %[[ADD3]], %[[ADD3]]
  func.func private @f(
      %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>,
      %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.add %0, %arg4 : tensor<4x8xf32>
    %3 = stablehlo.add %1, %2 : tensor<4x8xf32>
    return %3, %3, %3, %3
        : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @unused_duplicate_argument_avoids_cloning
module @unused_duplicate_argument_avoids_cloning {
  func.func @main(%arg0: !array, %arg1: !array) -> (!array, !array)
      attributes {ifrt.function} {
    // %arg1 is unused inside @f, so passing (%arg0, %arg0) requires no block
    // argument redirection and must not clone the shared public callee @f.
    // CHECK: %[[OUT0:.*]], %{{.*}} = ifrt.Call @f(%arg0, %arg0) on devices [0, 1]
    %0, %ctrl_0 = ifrt.Call @f(%arg0, %arg0) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: %[[OUT1:.*]], %{{.*}} = ifrt.Call @f(%arg0, %arg1) on devices [0, 1]
    %1, %ctrl_1 = ifrt.Call @f(%arg0, %arg1) on devices [0, 1]
        : (!array, !array) -> !array
    // CHECK: return %[[OUT0]], %[[OUT1]]
    return %0, %1 : !array, !array
  }

  // CHECK-LABEL: func.func @f(
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD]]
  // CHECK-NOT: func.func {{.*}}@f_
  func.func @f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

!array = !ifrt.array<tensor<4x8xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>

module @deeply_nested_callee_emits_error {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op cannot clone callee @outer::@inner::@main because only top-level functions and functions nested one module deep are supported}}
    %0, %ctrl_0 = ifrt.Call @outer::@inner::@main(%arg0, %arg0)
        on devices [0, 1] : (!array, !array) -> !array
    return %0 : !array
  }

  module @outer <sym_visibility = "private"> {
    module @inner {
      func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
          -> tensor<4x8xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
        return %0 : tensor<4x8xf32>
      }
    }
  }
}

