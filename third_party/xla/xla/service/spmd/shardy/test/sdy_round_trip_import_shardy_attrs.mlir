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
// =============================================================================

// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-import-shardy-attrs 2>&1 | FileCheck %s

// CHECK-LABEL: module @sparse_offload_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[BARRIER:.*]] = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%[[BARRIER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_sharded_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_sharded_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_sharded(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_sharded(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee_sharded(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_sharded(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_sharded_caller_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_sharded_caller_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_sharded(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[BARRIER:.*]] = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%[[BARRIER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_sharded(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee_sharded(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32> {
    mhlo.frontend_attributes = {
      xla.sdy.sharding = "#sdy.sharding<@mesh, [{\"a\"}, {}]>"
    }
  }) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_sharded(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_multi_args
module @sparse_offload_multi_args attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {
  // CHECK-LABEL: func private @sparse_offload_callee_multi_args(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[BARRIER:.*]] = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%[[BARRIER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_multi_args(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_multi_args(%arg0, %arg1) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @non_offloaded_call
module @non_offloaded_call attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {
  // CHECK-LABEL: func private @non_offloaded_callee(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @non_offloaded_callee(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @non_offloaded_callee(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @external_offloaded_call
module @external_offloaded_call {
  // CHECK-LABEL: func private @external_callee(tensor<8x8xf32>) -> tensor<8x8xf32>
  func.func private @external_callee(tensor<8x8xf32>) -> tensor<8x8xf32>

  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @external_callee(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_unreduced_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_unreduced_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_unreduced(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NOT:     sdy.propagation_barrier
  // CHECK:         %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a"}>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_unreduced(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={\"a\"}>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee_unreduced(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_unreduced(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_unreduced_max_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_unreduced_max_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_unreduced_max(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK:         %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced=max{"a"}>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_unreduced_max(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced=max{\"a\"}>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @x64_combine_flat_tuple_func_result_sharding
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @x64_combine_flat_tuple_func_result_sharding attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {
  // CHECK-LABEL: func @func_result_sharding_used_by_x64_combine_tuple(
  // CHECK-SAME:      %arg0: tensor<16xi64>, %arg1: tensor<16xi64>)
  // CHECK-SAME:      -> (tensor<16xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:          tensor<16xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}) {
  func.func @func_result_sharding_used_by_x64_combine_tuple(
      %arg0: tensor<16xi64>, %arg1: tensor<16xi64>) -> (tensor<16xi64>, tensor<16xi64>) {
    // CHECK-NEXT: %[[SPLIT_LOW_0:.*]] = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_HIGH_0:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_LOW_1:.*]] = stablehlo.custom_call @X64SplitLow(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_HIGH_1:.*]] = stablehlo.custom_call @X64SplitHigh(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[TUPLE:.*]] = stablehlo.tuple %[[SPLIT_LOW_0]], %[[SPLIT_HIGH_0]], %[[SPLIT_LOW_1]], %[[SPLIT_HIGH_1]]
    // CHECK-NEXT: %[[GTE_0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0]
    // CHECK-NEXT: %[[GTE_1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1]
    // CHECK-NEXT: %[[COMBINE_0:.*]] = stablehlo.custom_call @X64Combine(%[[GTE_0]], %[[GTE_1]])
    // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    // CHECK-NEXT: %[[GTE_2:.*]] = stablehlo.get_tuple_element %[[TUPLE]][2]
    // CHECK-NEXT: %[[GTE_3:.*]] = stablehlo.get_tuple_element %[[TUPLE]][3]
    // CHECK-NEXT: %[[COMBINE_1:.*]] = stablehlo.custom_call @X64Combine(%[[GTE_2]], %[[GTE_3]])
    // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    // CHECK-NEXT: return %[[COMBINE_0]], %[[COMBINE_1]]
    %0 = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    %1 = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    %2 = stablehlo.custom_call @X64SplitLow(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    %3 = stablehlo.custom_call @X64SplitHigh(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    %4 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%0)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %5 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%1)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %6 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%2)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"b\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %7 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%3)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"b\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %8 = stablehlo.tuple %4, %5, %6, %7 : tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>
    %9 = stablehlo.get_tuple_element %8[0] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %10 = stablehlo.get_tuple_element %8[1] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %11 = stablehlo.custom_call @X64Combine(%9, %10) : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    %12 = stablehlo.get_tuple_element %8[2] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %13 = stablehlo.get_tuple_element %8[3] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %14 = stablehlo.custom_call @X64Combine(%12, %13) : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    return %11, %14 : tensor<16xi64>, tensor<16xi64>
  }

  // CHECK-LABEL: func @func_result_sharding_used_by_x64_combine_multi_hop_tuple(
  // CHECK-SAME:      %arg0: tensor<16xi64>, %arg1: tensor<16xi64>)
  // CHECK-SAME:      -> (tensor<16xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:          tensor<16xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}) {
  func.func @func_result_sharding_used_by_x64_combine_multi_hop_tuple(
      %arg0: tensor<16xi64>, %arg1: tensor<16xi64>) -> (tensor<16xi64>, tensor<16xi64>) {
    // CHECK-NEXT: %[[SPLIT_LOW_0:.*]] = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_HIGH_0:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_LOW_1:.*]] = stablehlo.custom_call @X64SplitLow(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_HIGH_1:.*]] = stablehlo.custom_call @X64SplitHigh(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[T0:.*]] = stablehlo.tuple %[[SPLIT_LOW_0]], %[[SPLIT_HIGH_0]]
    // CHECK-NEXT: %[[T1:.*]] = stablehlo.tuple %[[SPLIT_LOW_1]], %[[SPLIT_HIGH_1]]
    // CHECK-NEXT: %[[T0_0:.*]] = stablehlo.get_tuple_element %[[T0]][0]
    // CHECK-NEXT: %[[T0_1:.*]] = stablehlo.get_tuple_element %[[T0]][1]
    // CHECK-NEXT: %[[T1_0:.*]] = stablehlo.get_tuple_element %[[T1]][0]
    // CHECK-NEXT: %[[T1_1:.*]] = stablehlo.get_tuple_element %[[T1]][1]
    // CHECK-NEXT: %[[FLAT_TUPLE:.*]] = stablehlo.tuple %[[T0_0]], %[[T0_1]], %[[T1_0]], %[[T1_1]]
    // CHECK-NEXT: %[[GTE_0:.*]] = stablehlo.get_tuple_element %[[FLAT_TUPLE]][0]
    // CHECK-NEXT: %[[GTE_1:.*]] = stablehlo.get_tuple_element %[[FLAT_TUPLE]][1]
    // CHECK-NEXT: %[[COMBINE_0:.*]] = stablehlo.custom_call @X64Combine(%[[GTE_0]], %[[GTE_1]])
    // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    // CHECK-NEXT: %[[GTE_2:.*]] = stablehlo.get_tuple_element %[[FLAT_TUPLE]][2]
    // CHECK-NEXT: %[[GTE_3:.*]] = stablehlo.get_tuple_element %[[FLAT_TUPLE]][3]
    // CHECK-NEXT: %[[COMBINE_1:.*]] = stablehlo.custom_call @X64Combine(%[[GTE_2]], %[[GTE_3]])
    // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    // CHECK-NEXT: return %[[COMBINE_0]], %[[COMBINE_1]]
    %0 = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    %1 = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    %2 = stablehlo.custom_call @X64SplitLow(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    %3 = stablehlo.custom_call @X64SplitHigh(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    %4 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%0)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %5 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%1)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %6 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%2)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"b\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %7 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%3)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"b\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %8 = stablehlo.tuple %4, %5 : tuple<tensor<16xui32>, tensor<16xui32>>
    %9 = stablehlo.tuple %6, %7 : tuple<tensor<16xui32>, tensor<16xui32>>
    %10 = stablehlo.get_tuple_element %8[0] : (tuple<tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %11 = stablehlo.get_tuple_element %8[1] : (tuple<tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %12 = stablehlo.get_tuple_element %9[0] : (tuple<tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %13 = stablehlo.get_tuple_element %9[1] : (tuple<tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %14 = stablehlo.tuple %10, %11, %12, %13 : tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>
    %15 = stablehlo.get_tuple_element %14[0] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %16 = stablehlo.get_tuple_element %14[1] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %17 = stablehlo.custom_call @X64Combine(%15, %16) : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    %18 = stablehlo.get_tuple_element %14[2] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %19 = stablehlo.get_tuple_element %14[3] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %20 = stablehlo.custom_call @X64Combine(%18, %19) : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    return %17, %20 : tensor<16xi64>, tensor<16xi64>
  }

  // CHECK-LABEL: func @func_result_sharding_used_by_x64_combine_nested_tuple(
  // CHECK-SAME:      %arg0: tensor<16xi64>, %arg1: tensor<16xi64>)
  // CHECK-SAME:      -> (tensor<16xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:          tensor<16xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}) {
  func.func @func_result_sharding_used_by_x64_combine_nested_tuple(
      %arg0: tensor<16xi64>, %arg1: tensor<16xi64>) -> (tensor<16xi64>, tensor<16xi64>) {
    // CHECK-NEXT: %[[SPLIT_LOW_0:.*]] = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_HIGH_0:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_LOW_1:.*]] = stablehlo.custom_call @X64SplitLow(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[SPLIT_HIGH_1:.*]] = stablehlo.custom_call @X64SplitHigh(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    // CHECK-NEXT: %[[INNER:.*]] = stablehlo.tuple %[[SPLIT_LOW_0]], %[[SPLIT_HIGH_0]], %[[SPLIT_LOW_1]], %[[SPLIT_HIGH_1]]
    // CHECK-NEXT: %[[OUTER:.*]] = stablehlo.tuple %[[INNER]]
    // CHECK-NEXT: %[[OUTER_0:.*]] = stablehlo.get_tuple_element %[[OUTER]][0]
    // CHECK-NEXT: %[[GTE_0:.*]] = stablehlo.get_tuple_element %[[OUTER_0]][0]
    // CHECK-NEXT: %[[GTE_1:.*]] = stablehlo.get_tuple_element %[[OUTER_0]][1]
    // CHECK-NEXT: %[[COMBINE_0:.*]] = stablehlo.custom_call @X64Combine(%[[GTE_0]], %[[GTE_1]])
    // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    // CHECK-NEXT: %[[GTE_2:.*]] = stablehlo.get_tuple_element %[[OUTER_0]][2]
    // CHECK-NEXT: %[[GTE_3:.*]] = stablehlo.get_tuple_element %[[OUTER_0]][3]
    // CHECK-NEXT: %[[COMBINE_1:.*]] = stablehlo.custom_call @X64Combine(%[[GTE_2]], %[[GTE_3]])
    // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    // CHECK-NEXT: return %[[COMBINE_0]], %[[COMBINE_1]]
    %0 = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    %1 = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<16xi64>) -> tensor<16xui32>
    %2 = stablehlo.custom_call @X64SplitLow(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    %3 = stablehlo.custom_call @X64SplitHigh(%arg1) : (tensor<16xi64>) -> tensor<16xui32>
    %4 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%0)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %5 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%1)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %6 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%2)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"b\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %7 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%3)
        {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"b\"}]>]>"}} :
        (tensor<16xui32>) -> tensor<16xui32>
    %8 = stablehlo.tuple %4, %5, %6, %7 : tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>
    %9 = stablehlo.tuple %8 : tuple<tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>>
    %10 = stablehlo.get_tuple_element %9[0] : (tuple<tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>>) -> tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>
    %11 = stablehlo.get_tuple_element %10[0] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %12 = stablehlo.get_tuple_element %10[1] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %13 = stablehlo.custom_call @X64Combine(%11, %12) : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    %14 = stablehlo.get_tuple_element %10[2] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %15 = stablehlo.get_tuple_element %10[3] : (tuple<tensor<16xui32>, tensor<16xui32>, tensor<16xui32>, tensor<16xui32>>) -> tensor<16xui32>
    %16 = stablehlo.custom_call @X64Combine(%14, %15) : (tensor<16xui32>, tensor<16xui32>) -> tensor<16xi64>
    return %13, %16 : tensor<16xi64>, tensor<16xi64>
  }
}

