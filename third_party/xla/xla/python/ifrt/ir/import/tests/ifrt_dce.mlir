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
// RUN: ifrt-opt %s -ifrt-dce -split-input-file | FileCheck %s

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @unused_operand
module @unused_operand {
  // CHECK: func.func @main(%arg0: !ifrt.array<{{.*}}>, %arg1: !ifrt.array<{{.*}}>)
  func.func @main(%arg0: !array0, %arg1: !array0) -> !array0
      attributes {ifrt.function} {
    // CHECK-NEXT: %[[OUT:.*]], %{{.*}} = ifrt.Call @f(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Call @f(%arg0, %arg1) on devices [0, 1, 2, 3]
        : (!array0, !array0) -> !array0
    return %0 : !array0
  }

  // CHECK: func.func private @f(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  func.func private @f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @unused_result
module @unused_result {
  func.func @main(%arg0: !array0) -> !array0 attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @f(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0:2, %ctrl_0 = ifrt.Call @f(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> (!array0, !array0)
    return %0#0 : !array0
  }

  // CHECK: func.func private @f(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  func.func private @f(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @unused_result_causes_operand_to_be_removed
module @unused_result_causes_operand_to_be_removed {
  func.func @main(%arg0: !array0) -> !array0 attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @producer
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @consumer(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    %1:2, %ctrl_1 = ifrt.Call @consumer(%arg0, %0) on devices [0, 1, 2, 3]
        : (!array0, !array0) -> (!array0, !array0)
    return %1#0 : !array0
  }

  // CHECK-NOT: func.func private @producer
  func.func private @producer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }

  // CHECK: func.func private @consumer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  func.func private @consumer(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @pure_ops_are_removed_but_side_effecting_ops_are_not
module @pure_ops_are_removed_but_side_effecting_ops_are_not {
  func.func @main(%arg0: !array0) -> !array0 attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @f(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0:3, %ctrl_0 = ifrt.Call @f(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> (!array0, !array0, !array0)
    return %0#2 : !array0
  }

  // CHECK: func.func private @f(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %{{.*}} = stablehlo.custom_call @Sharding(%arg0) {has_side_effect = true}
  // CHECK-NEXT: return %arg0
  func.func private @f(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.custom_call @Sharding(%arg0) {has_side_effect = true}
        : (tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0, %1, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>
!array1 = !ifrt.array<tensor<4x8xf32>, #sharding, [4, 5, 6, 7]>

// CHECK-LABEL: module @reshard_becomes_unused
module @reshard_becomes_unused {
  func.func @main(%arg0: !array0, %arg1: !array1) -> !array1
      attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @producer
    // CHECK-NOT: ifrt.Reshard
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @consumer(%arg1) on devices [4, 5, 6, 7] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.Reshard(%0) : (!array0) -> !array1
    %2:2, %ctrl_2 = ifrt.Call @consumer(%arg1, %1) on devices [4, 5, 6, 7]
        : (!array1, !array1) -> (!array1, !array1)
    return %2#0 : !array1
  }

  // CHECK-NOT: func.func private @producer
  func.func private @producer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }

  // CHECK: func.func private @consumer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  func.func private @consumer(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>
!array1 = !ifrt.array<tensor<4x8xf32>, #sharding, [4, 5, 6, 7]>

// CHECK-LABEL: module @copy_arrays_becomes_unused
module @copy_arrays_becomes_unused {
  func.func @main(%arg0: !array0, %arg1: !array1) -> !array1
      attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @producer
    // CHECK-NOT: ifrt.CopyArrays
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @consumer(%arg1) on devices [4, 5, 6, 7] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array0) -> !array1
    %2:2, %ctrl_2 = ifrt.Call @consumer(%arg1, %1) on devices [4, 5, 6, 7]
        : (!array1, !array1) -> (!array1, !array1)
    return %2#0 : !array1
  }

  // CHECK-NOT: func.func private @producer
  func.func private @producer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }

  // CHECK: func.func private @consumer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  func.func private @consumer(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>
!array1 = !ifrt.array<tensor<2x16xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @bitcast_arrays_becomes_unused
module @bitcast_arrays_becomes_unused {
  func.func @main(%arg0: !array0, %arg1: !array1) -> !array1
      attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @producer
    // CHECK-NOT: ifrt.BitcastArrays
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @consumer(%arg1) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.BitcastArrays(%0) <donated = true> : (!array0) -> !array1
    %2:2, %ctrl_2 = ifrt.Call @consumer(%arg1, %1) on devices [0, 1, 2, 3]
        : (!array1, !array1) -> (!array1, !array1)
    return %2#0 : !array1
  }

  // CHECK-NOT: func.func private @producer
  func.func private @producer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }

  // CHECK: func.func private @consumer(%arg0: tensor<2x16xf32>) -> tensor<2x16xf32>
  func.func private @consumer(%arg0: tensor<2x16xf32>, %arg1: tensor<2x16xf32>)
      -> (tensor<2x16xf32>, tensor<2x16xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x16xf32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<2x16xf32>
    return %0, %1 : tensor<2x16xf32>, tensor<2x16xf32>
  }
}

// -----

#sharding0 = #ifrt.sharding_param<2x2 to [0] on 4>
#sharding1 = #ifrt.sharding_param<4x1 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding0, [0, 1, 2, 3]>
!array1 = !ifrt.array<tensor<4x8xf32>, #sharding1, [4, 5, 6, 7]>
!array2 = !ifrt.array<tensor<4x8xf32>, #sharding0, [4, 5, 6, 7]>

// CHECK-LABEL: module @partial_reshard_and_copy_arrays_dce
module @partial_reshard_and_copy_arrays_dce {
  func.func @main(%arg0: !array0, %arg1: !array0) -> (!array1, !array2)
      attributes {ifrt.function} {
    // CHECK: %[[R:.*]], %{{.*}} = ifrt.Reshard(%arg1) {ifrt.dst_mesh_name = "m2", ifrt.src_mesh_name = "m1"} : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[C:.*]], %{{.*}} = ifrt.CopyArrays(%arg0) : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[R]], %[[C]]
    %0:2, %ctrl_0 = ifrt.Reshard(%arg0, %arg1)
        {ifrt.src_mesh_name = "m1", ifrt.dst_mesh_name = "m2"}
        : (!array0, !array0) -> (!array1, !array1)
    %1:2, %ctrl_1 = ifrt.CopyArrays(%arg0, %arg1)
        : (!array0, !array0) -> (!array2, !array2)
    return %0#1, %1#0 : !array1, !array2
  }
}

// -----

#sharding0 = #ifrt.sharding_param<2x2 to [0] on 4>
#sharding1 = #ifrt.sharding_param<4x1 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding0, [0, 1, 2, 3]>
!array1 = !ifrt.array<tensor<4x8xf32>, #sharding1, [4, 5, 6, 7]>
!array2 = !ifrt.array<tensor<4x8xf32>, #sharding0, [4, 5, 6, 7]>
!array_bc = !ifrt.array<tensor<2x16xf32>, #sharding0, [0, 1, 2, 3]>

// CHECK-LABEL: module @donated_pairwise_ops_dce
module @donated_pairwise_ops_dce {
  func.func @main(%arg0: !array0 {ifrt.donated}, %arg1: !array0 {ifrt.donated},
                  %arg2: !array0 {ifrt.donated}, %arg3: !array0 {ifrt.donated},
                  %arg4: !array0 {ifrt.donated}, %arg5: !array0 {ifrt.donated},
                  %arg6: !array0 {ifrt.donated}, %arg7: !array0 {ifrt.donated},
                  %arg8: !array0 {ifrt.donated})
      -> (!array1, !array2, !array_bc) attributes {ifrt.function} {
    // CHECK: func.func @main(
    // CHECK-NEXT: %[[R:.*]], %[[CTRL_R:.*]] = ifrt.Reshard(%arg4) <donated = true> : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[C:.*]], %{{.*}} = ifrt.CopyArrays(%arg5) <donated = true> : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[BC:.*]], %{{.*}} = ifrt.BitcastArrays(%arg8) after %[[CTRL_R]] <donated = true> {test.tag = "bc"} : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[R]], %[[C]], %[[BC]]
    %0, %ctrl_0 = ifrt.Reshard(%arg0) <donated = true> : (!array0) -> !array1
    %1, %ctrl_1 = ifrt.CopyArrays(%arg1) <donated = true> : (!array0) -> !array2
    %2, %ctrl_2 = ifrt.BitcastArrays(%arg2) <donated = true>
        : (!array0) -> !array_bc
    %3:2, %ctrl_3 = ifrt.Reshard(%arg3, %arg4) <donated = true>
        : (!array0, !array0) -> (!array1, !array1)
    %4:2, %ctrl_4 = ifrt.CopyArrays(%arg5, %arg6) <donated = true>
        : (!array0, !array0) -> (!array2, !array2)
    %5:2, %ctrl_5 = ifrt.BitcastArrays(%arg7, %arg8) after %ctrl_3
        <donated = true> {test.tag = "bc"}
        : (!array0, !array0) -> (!array_bc, !array_bc)
    return %3#1, %4#0, %5#1 : !array1, !array2, !array_bc
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>
!array1 = !ifrt.array<tensor<4x8xf32>, #sharding, [4, 5, 6, 7]>

// CHECK-LABEL: module @copy_arrays_reuse_dce
module @copy_arrays_reuse_dce {
  func.func @main(%arg0: !array0, %arg1: !array0, %arg2: !array0) -> !array1
      attributes {ifrt.function} {
    // CHECK: func.func @main(
    // CHECK-NEXT: %[[C:.*]], %{{.*}} = ifrt.CopyArrays(%arg2) <reuse = true> : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[C]]
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse = true> : (!array0) -> !array1
    %1:2, %ctrl_1 = ifrt.CopyArrays(%arg1, %arg2) <reuse = true>
        : (!array0, !array0) -> (!array1, !array1)
    return %1#1 : !array1
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @remap_io_aliases_donations_and_attrs
module @remap_io_aliases_donations_and_attrs {
  func.func @main(%arg0: !array0 {ifrt.donated}, %arg1: !array0 {ifrt.donated},
                  %arg2: !array0 {ifrt.donated}, %arg3: !array0 {ifrt.donated})
      -> (!array0, !array0) attributes {ifrt.function} {
    // Input 0 (donated) and input 1 (aliased to output 0) are unused after
    // output 0 is erased; input 2 (donated) -> new input 0, input 3 (aliased
    // to output 2) -> new input 1, output 1 -> new output 0, output 2 -> new
    // output 1.
    // CHECK: %[[OUT:.*]]:2, %{{.*}} = ifrt.Call @f(%arg2, %arg3) on devices [0, 1, 2, 3]
    // CHECK-SAME: <arg_attrs = [{test.arg = "in2"}, {test.arg = "in3"}],
    // CHECK-SAME:  res_attrs = [{test.res = "out1"}, {test.res = "out2"}],
    // CHECK-SAME:  io_aliases = [array<i32: 1, 1>],
    // CHECK-SAME:  donated_input_indices = [0]>
    // CHECK-SAME: {ifrt.origin = ["stage0"]}
    // CHECK-NEXT: return %[[OUT]]#0, %[[OUT]]#1
    %0:3, %ctrl_0 = ifrt.Call @f(%arg0, %arg1, %arg2, %arg3)
        on devices [0, 1, 2, 3]
        <arg_attrs = [{test.arg = "in0"}, {test.arg = "in1"},
                      {test.arg = "in2"}, {test.arg = "in3"}],
         donated_input_indices = [0, 2],
         io_aliases = [array<i32: 1, 0>, array<i32: 3, 2>],
         res_attrs = [{test.res = "out0"}, {test.res = "out1"},
                      {test.res = "out2"}]>
        {ifrt.origin = ["stage0"]}
        : (!array0, !array0, !array0, !array0) -> (!array0, !array0, !array0)
    return %0#1, %0#2 : !array0, !array0
  }

  // CHECK: func.func private @f(
  // CHECK-SAME: %arg0: tensor<4x8xf32> {jax.buffer_donor = true, test.callee_arg = "a2"},
  // CHECK-SAME: %arg1: tensor<4x8xf32> {test.callee_arg = "a3", tf.aliasing_output = 1 : i32})
  // CHECK-SAME: -> (tensor<4x8xf32> {test.callee_res = "r1"}, tensor<4x8xf32> {test.callee_res = "r2"})
  func.func private @f(
      %arg0: tensor<4x8xf32> {jax.buffer_donor = true, test.callee_arg = "a0"},
      %arg1: tensor<4x8xf32> {test.callee_arg = "a1", tf.aliasing_output = 0 : i32},
      %arg2: tensor<4x8xf32> {jax.buffer_donor = true, test.callee_arg = "a2"},
      %arg3: tensor<4x8xf32> {test.callee_arg = "a3", tf.aliasing_output = 2 : i32})
      -> (tensor<4x8xf32> {test.callee_res = "r0"},
          tensor<4x8xf32> {test.callee_res = "r1"},
          tensor<4x8xf32> {test.callee_res = "r2"}) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg3, %arg3 : tensor<4x8xf32>
    return %0, %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @zero_output_side_effecting_call_preserved
module @zero_output_side_effecting_call_preserved {
  func.func @main(%arg0: !array0) attributes {ifrt.function} {
    // CHECK: %{{.*}} = ifrt.Call @side_effect(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> ()
    %0, %ctrl_0 = ifrt.Call @side_effect(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    return
  }

  // CHECK: func.func private @side_effect(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT: %{{.*}} = stablehlo.custom_call @SideEffect(%arg0) {has_side_effect = true}
  // CHECK-NEXT: return
  func.func private @side_effect(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.custom_call @SideEffect(%arg0) {has_side_effect = true}
        : (tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @shared_callee_cloned_on_partial_dce
module @shared_callee_cloned_on_partial_dce {
  func.func @main(%arg0: !array0) -> (!array0, !array0, !array0)
      attributes {ifrt.function} {
    // CHECK: %[[FULL:.*]]:2, %{{.*}} = ifrt.Call @shared(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> (!ifrt.array<{{.*}}>, !ifrt.array<{{.*}}>)
    // CHECK-NEXT: %[[PARTIAL:.*]], %{{.*}} = ifrt.Call @[[CLONED:shared_.+]](%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[FULL]]#0, %[[FULL]]#1, %[[PARTIAL]]
    %0:2, %ctrl_0 = ifrt.Call @shared(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> (!array0, !array0)
    %1:2, %ctrl_1 = ifrt.Call @shared(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> (!array0, !array0)
    return %0#0, %0#1, %1#0 : !array0, !array0, !array0
  }

  // CHECK: func.func private @shared(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  // CHECK: func.func private @[[CLONED]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  func.func private @shared(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf32>
    return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2x2 to [0] on 4>
!array0 = !ifrt.array<tensor<4x8xf32>, #sharding, [0, 1, 2, 3]>

// CHECK-LABEL: module @shared_callee_one_dead_one_live
module @shared_callee_one_dead_one_live {
  func.func @main(%arg0: !array0) -> !array0 attributes {ifrt.function} {
    // CHECK: %[[LIVE:.*]], %{{.*}} = ifrt.Call @shared_one(%arg0) on devices [0, 1, 2, 3] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[LIVE]]
    %0, %ctrl_0 = ifrt.Call @shared_one(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.Call @shared_one(%arg0) on devices [0, 1, 2, 3]
        : (!array0) -> !array0
    return %0 : !array0
  }

  // CHECK: func.func private @shared_one(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NOT: func.func private @shared_one_
  func.func private @shared_one(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

!array_d0 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d1 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_d01 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @remap_arrays_becomes_unused
module @remap_arrays_becomes_unused {
  func.func @main(%arg0: !array_d0, %arg1: !array_d1, %arg2: !array_d01)
      -> !array_d01 attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @prod0
    // CHECK-NOT: ifrt.Call @prod1
    // CHECK-NOT: ifrt.RemapArrays
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @consumer(%arg2) on devices [0, 1] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Call @prod0(%arg0) on devices [0]
        : (!array_d0) -> !array_d0
    %1, %ctrl_1 = ifrt.Call @prod1(%arg1) on devices [1]
        : (!array_d1) -> !array_d1
    %2, %ctrl_2 = ifrt.RemapArrays(%0, %1)
        mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                    #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
        <donated = true>
        : (!array_d0, !array_d1) -> !array_d01
    %3:2, %ctrl_3 = ifrt.Call @consumer(%arg2, %2) on devices [0, 1]
        : (!array_d01, !array_d01) -> (!array_d01, !array_d01)
    return %3#0 : !array_d01
  }

  // CHECK-NOT: func.func private @prod0
  func.func private @prod0(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }

  // CHECK-NOT: func.func private @prod1
  func.func private @prod1(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @consumer(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32>
  func.func private @consumer(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>)
      -> (tensor<2x4xi32>, tensor<2x4xi32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x4xi32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<2x4xi32>
    return %0, %1 : tensor<2x4xi32>, tensor<2x4xi32>
  }
}

// -----

!array_4dev = !ifrt.array<tensor<2x8xi32>,
                          #ifrt.sharding_param<1x4 to [0] on 4>, [0, 1, 2, 3]>
!array_dev01 = !ifrt.array<tensor<2x4xi32>,
                           #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>
!array_dev2 = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
!array_dev3 = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 1>, [3]>

// CHECK-LABEL: module @partial_remap_arrays_fan_out_dce
module @partial_remap_arrays_fan_out_dce {
  func.func @main(%arg0: !array_4dev) -> (!array_dev01, !array_dev3)
      attributes {ifrt.function} {
    // CHECK: %[[OUT:.*]]:2, %{{.*}} = ifrt.RemapArrays(%arg0)
    // CHECK-SAME: mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0 : 2 : 1] to [0 : 2 : 1]>]>, #ifrt.array_mapping<0, 1, [#ifrt.mapping<[3 : 4 : 1] to [0 : 1 : 1]>]>]
    // CHECK-SAME: {test.tag = "fan_out"}
    // CHECK-NEXT: return %[[OUT]]#0, %[[OUT]]#1
    %0:3, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings = [
          #ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:2:1] to [0:2:1]>]>,
          #ifrt.array_mapping<0, 1, [#ifrt.mapping<[2:3:1] to [0:1:1]>]>,
          #ifrt.array_mapping<0, 2, [#ifrt.mapping<[3:4:1] to [0:1:1]>]>
        ]
        {test.tag = "fan_out"}
        : (!array_4dev) -> (!array_dev01, !array_dev2, !array_dev3)
    return %0#0, %0#2 : !array_dev01, !array_dev3
  }
}

// -----

!array_d0 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d1 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_d2 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
!array_d01 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>
!array_d23 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x2 to [0] on 2>, [2, 3]>

// CHECK-LABEL: module @partial_remap_arrays_m_to_n_and_fan_in_prunes_inputs
module @partial_remap_arrays_m_to_n_and_fan_in_prunes_inputs {
  func.func @main(%arg0: !array_d0, %arg1: !array_d1, %arg2: !array_d23)
      -> !array_d01 attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @dead_prod0
    // CHECK-NOT: ifrt.Call @dead_prod3
    // CHECK: %[[P1:.*]], %{{.*}} = ifrt.Call @live_prod1(%arg0) on devices [0] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[P2:.*]], %[[CTRL2:.*]] = ifrt.Call @live_prod2(%arg1) on devices [1] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[REMAP:.*]], %{{.*}} = ifrt.RemapArrays(%[[P1]], %[[P2]])
    // CHECK-SAME: mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0 : 1 : 1] to [0 : 1 : 1]>]>, #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0 : 1 : 1] to [1 : 2 : 1]>]>]
    // CHECK-SAME: after %[[CTRL2]] <donated = true> {test.tag = "m_to_n"}
    // CHECK-NEXT: return %[[REMAP]]
    %p0, %ctrl_0 = ifrt.Call @dead_prod0(%arg0) on devices [0]
        : (!array_d0) -> !array_d0
    %p1, %ctrl_1 = ifrt.Call @live_prod1(%arg0) on devices [0]
        : (!array_d0) -> !array_d0
    %p2:2, %ctrl_2 = ifrt.Call @live_prod2(%arg1) on devices [1]
        : (!array_d1) -> (!array_d1, !array_d1)
    %p3, %ctrl_3 = ifrt.Call @dead_prod3(%arg2) on devices [2, 3]
        : (!array_d23) -> !array_d23
    %r:3, %ctrl_r = ifrt.RemapArrays(%p0, %p1, %p2#0, %p3)
        mappings = [
          #ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
          #ifrt.array_mapping<1, 1, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
          #ifrt.array_mapping<2, 1, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>,
          #ifrt.array_mapping<3, 2, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>
        ]
        after %ctrl_2
        <donated = true>
        {test.tag = "m_to_n"}
        : (!array_d0, !array_d0, !array_d1, !array_d23)
          -> (!array_d0, !array_d01, !array_d2)
    return %r#1 : !array_d01
  }

  // CHECK-NOT: func.func private @dead_prod0
  func.func private @dead_prod0(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @live_prod1(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32>
  func.func private @live_prod1(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @live_prod2(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32>
  func.func private @live_prod2(%arg0: tensor<2x2xi32>)
      -> (tensor<2x2xi32>, tensor<2x2xi32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xi32>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<2x2xi32>
    return %0, %1 : tensor<2x2xi32>, tensor<2x2xi32>
  }

  // CHECK-NOT: func.func private @dead_prod3
  func.func private @dead_prod3(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }
}

// -----

!array_d01_i32 = !ifrt.array<tensor<2x4xi32>,
                             #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>
!array_d0_i32 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d1_i32 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array_d0_1x2x2 = !ifrt.array<tensor<1x2x2xi32>,
                              #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array_d1_1x2x2 = !ifrt.array<tensor<1x2x2xi32>,
                              #ifrt.sharding_param<1x1x1 to [0] on 1>, [1]>
!array_d2_1x2x2 = !ifrt.array<tensor<1x2x2xi32>,
                              #ifrt.sharding_param<1x1x1 to [0] on 1>, [2]>
!array_d3_1x2x2 = !ifrt.array<tensor<1x2x2xi32>,
                              #ifrt.sharding_param<1x1x1 to [0] on 1>, [3]>

// CHECK-LABEL: module @chained_call_remap_bitcast_copy_reshard_dce
module @chained_call_remap_bitcast_copy_reshard_dce {
  func.func @main(%arg0: !array_d01_i32, %arg1: !array_d0_i32) -> !array_d2_1x2x2
      attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @dead_chain_prod
    // CHECK-NOT: ifrt.RemapArrays
    // CHECK: %[[LIVE:.*]], %{{.*}} = ifrt.Call @live_chain_prod(%arg1) on devices [0] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[BC:.*]], %{{.*}} = ifrt.BitcastArrays(%[[LIVE]]) <donated = true> : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[CP:.*]], %{{.*}} = ifrt.CopyArrays(%[[BC]]) : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: %[[RS:.*]], %{{.*}} = ifrt.Reshard(%[[CP]]) : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[RS]]
    %dead_src, %ctrl_0 = ifrt.Call @dead_chain_prod(%arg0) on devices [0, 1]
        : (!array_d01_i32) -> !array_d01_i32
    %live_src, %ctrl_1 = ifrt.Call @live_chain_prod(%arg1) on devices [0]
        : (!array_d0_i32) -> !array_d0_i32
    %remapped:2, %ctrl_2 = ifrt.RemapArrays(%dead_src)
        mappings = [
          #ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
          #ifrt.array_mapping<0, 1, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>
        ]
        : (!array_d01_i32) -> (!array_d0_i32, !array_d1_i32)
    %bc:2, %ctrl_3 = ifrt.BitcastArrays(%live_src, %remapped#0) <donated = true>
        : (!array_d0_i32, !array_d0_i32) -> (!array_d0_1x2x2, !array_d0_1x2x2)
    %dead_bc, %ctrl_4 = ifrt.BitcastArrays(%remapped#1) <donated = true>
        : (!array_d1_i32) -> !array_d1_1x2x2
    %cp:2, %ctrl_5 = ifrt.CopyArrays(%bc#0, %bc#1)
        : (!array_d0_1x2x2, !array_d0_1x2x2) -> (!array_d2_1x2x2, !array_d2_1x2x2)
    %rs:2, %ctrl_6 = ifrt.Reshard(%cp#0, %dead_bc)
        : (!array_d2_1x2x2, !array_d1_1x2x2) -> (!array_d2_1x2x2, !array_d3_1x2x2)
    return %rs#0 : !array_d2_1x2x2
  }

  // CHECK-NOT: func.func private @dead_chain_prod
  func.func private @dead_chain_prod(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }

  // CHECK: func.func private @live_chain_prod(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32>
  func.func private @live_chain_prod(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

!array_2x4 = !ifrt.array<tensor<2x4xi32>,
                         #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>
!array_2x4_d23 = !ifrt.array<tensor<2x4xi32>,
                             #ifrt.sharding_param<1x2 to [0] on 2>, [2, 3]>
!array_2x4_sharded0 = !ifrt.array<tensor<2x4xi32>,
                                  #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_1x2x4 = !ifrt.array<tensor<1x2x4xi32>,
                           #ifrt.sharding_param<1x1x2 to [0] on 2>, [0, 1]>
!array_2x2 = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 1>, [0]>

// CHECK-LABEL: module @ctrl_used_ops_without_ctrl_inputs_eliminated
module @ctrl_used_ops_without_ctrl_inputs_eliminated {
  func.func @main(%arg0: !array_2x4, %arg1: !array_2x4 {ifrt.donated})
      -> !array_2x4 attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Reshard
    // CHECK-NOT: ifrt.CopyArrays
    // CHECK-NOT: ifrt.BitcastArrays
    // CHECK-NOT: ifrt.RemapArrays
    // CHECK-NOT: ifrt.Call @dead_pure
    // CHECK: %[[OUT:.*]], %{{.*}} = ifrt.Call @id(%arg0) on devices [0, 1] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
    // CHECK-NEXT: return %[[OUT]]
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array_2x4) -> !array_2x4_sharded0
    %1, %ctrl_1 = ifrt.CopyArrays(%arg0) : (!array_2x4) -> !array_2x4_d23
    %2, %ctrl_2 = ifrt.BitcastArrays(%arg1) <donated = true>
        : (!array_2x4) -> !array_1x2x4
    %3, %ctrl_3 = ifrt.RemapArrays(%arg0)
        mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array_2x4) -> !array_2x2
    %4, %ctrl_4 = ifrt.Call @dead_pure(%arg0) on devices [0, 1]
        : (!array_2x4) -> !array_2x4
    %5, %ctrl_5 = ifrt.Call @id(%arg0)
        after %ctrl_0, %ctrl_1, %ctrl_2, %ctrl_3, %ctrl_4 on devices [0, 1]
        : (!array_2x4) -> !array_2x4
    return %5 : !array_2x4
  }

  // CHECK-NOT: func.func private @dead_pure
  func.func private @dead_pure(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }

  func.func private @id(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }
}

// -----

!array_d0 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d1 = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [1]>

// CHECK-LABEL: module @remap_arrays_unreferenced_input_pruned
module @remap_arrays_unreferenced_input_pruned {
  func.func @main(%arg0: !array_d0, %arg1: !array_d1) -> !array_d1
      attributes {ifrt.function} {
    // CHECK-NOT: ifrt.Call @dead_prod
    // CHECK: %[[RM:.*]], %{{.*}} = ifrt.RemapArrays(%arg1)
    // CHECK-SAME: mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0 : 1 : 1] to [0 : 1 : 1]>]>]
    // CHECK-NEXT: return %[[RM]]
    %dead, %ctrl_0 = ifrt.Call @dead_prod(%arg0) on devices [0]
        : (!array_d0) -> !array_d0
    %out, %ctrl_1 = ifrt.RemapArrays(%dead, %arg1)
        mappings = [#ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array_d0, !array_d1) -> !array_d1
    return %out : !array_d1
  }

  // CHECK-NOT: func.func private @dead_prod
  func.func private @dead_prod(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

!arr_01 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>
!arr_23 = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2, 3]>
!arr_rs = !ifrt.array<tensor<2x4xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!arr_bc = !ifrt.array<tensor<1x2x4xi32>,
                      #ifrt.sharding_param<1x1x2 to [0] on 2>, [0, 1]>
!arr_d0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>

// CHECK-LABEL: module @full_elimination_forwards_and_deduplicates_control_inputs
module @full_elimination_forwards_and_deduplicates_control_inputs {
  func.func @main(%arg0: !arr_01, %arg1: !arr_01 {ifrt.donated})
      -> (!arr_01, !arr_01, !arr_01, !arr_01) attributes {ifrt.function} {
    // CHECK: %[[A:.*]], %[[CTRL_A:.*]] = ifrt.Call @stage_a(%arg0) on devices [0, 1]
    // CHECK-NEXT: %[[B:.*]], %[[CTRL_B:.*]] = ifrt.Call @stage_b(%arg0) on devices [0, 1]
    // CHECK-NEXT: %[[C:.*]], %[[CTRL_C:.*]] = ifrt.Call @stage_c(%arg0) on devices [0, 1]
    // CHECK-NEXT: %[[CTRL_D:.*]] = "ifrt.After"(%[[A]]) : (!ifrt.array<{{.*}}>) -> !ifrt.control
    // CHECK-NOT: ifrt.CopyArrays
    // CHECK-NOT: ifrt.Reshard
    // CHECK-NOT: ifrt.BitcastArrays
    // CHECK-NOT: ifrt.RemapArrays
    // CHECK-NOT: ifrt.Call @dead_callee
    // CHECK: %[[LIVE:.*]], %{{.*}} = ifrt.Call @live(%[[A]], %[[B]], %[[C]]) after %[[CTRL_A]], %[[CTRL_B]], %[[CTRL_C]], %[[CTRL_D]] on devices [0, 1]
    // CHECK-NEXT: return %[[A]], %[[B]], %[[C]], %[[LIVE]]
    %a, %ctrl_a = ifrt.Call @stage_a(%arg0) on devices [0, 1]
        : (!arr_01) -> !arr_01
    %b, %ctrl_b = ifrt.Call @stage_b(%arg0) on devices [0, 1]
        : (!arr_01) -> !arr_01
    %c, %ctrl_c = ifrt.Call @stage_c(%arg0) on devices [0, 1]
        : (!arr_01) -> !arr_01
    %ctrl_d = "ifrt.After"(%a) : (!arr_01) -> !ifrt.control
    %dead_cp, %ctrl_cp = ifrt.CopyArrays(%arg0) after %ctrl_a, %ctrl_b
        : (!arr_01) -> !arr_23
    %dead_rs, %ctrl_rs = ifrt.Reshard(%arg0) after %ctrl_b, %ctrl_c
        : (!arr_01) -> !arr_rs
    %dead_bc, %ctrl_bc = ifrt.BitcastArrays(%arg1) after %ctrl_b, %ctrl_c
        <donated = true> : (!arr_01) -> !arr_bc
    %dead_rm, %ctrl_rm = ifrt.RemapArrays(%arg0)
        mappings = [#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        after %ctrl_a, %ctrl_d
        : (!arr_01) -> !arr_d0
    %dead_call, %ctrl_dead_call = ifrt.Call @dead_callee(%arg0)
        after %ctrl_c, %ctrl_d on devices [0, 1]
        : (!arr_01) -> !arr_01
    %live, %ctrl_live = ifrt.Call @live(%a, %b, %c)
        after %ctrl_a, %ctrl_cp, %ctrl_rs, %ctrl_bc, %ctrl_rm, %ctrl_dead_call
        on devices [0, 1]
        : (!arr_01, !arr_01, !arr_01) -> !arr_01
    return %a, %b, %c, %live : !arr_01, !arr_01, !arr_01, !arr_01
  }

  func.func private @stage_a(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }

  func.func private @stage_b(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }

  func.func private @stage_c(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }

  // CHECK-NOT: func.func private @dead_callee
  func.func private @dead_callee(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }

  func.func private @live(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>,
                          %arg2: tensor<2x4xi32>) -> tensor<2x4xi32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xi32>
    %1 = stablehlo.add %0, %arg2 : tensor<2x4xi32>
    return %1 : tensor<2x4xi32>
  }
}
