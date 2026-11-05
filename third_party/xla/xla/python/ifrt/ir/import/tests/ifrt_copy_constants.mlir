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
// RUN: ifrt-opt %s -ifrt-copy-constants -split-input-file | FileCheck %s

!array_m1 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @call_to_calls
module @call_to_calls {
  func.func @main() -> (!array_m1, !array_m1, !array_m1)
      attributes {ifrt.function} {
    // CHECK: %[[OUT0:.+]], %{{.+}} = ifrt.Call @f0()
    %0, %ctrl_0 = ifrt.Call @f0() on devices [0, 1] : () -> !array_m1
    // CHECK: %[[OUT1:.+]], %{{.+}} = ifrt.Call @f1(%[[OUT0]])
    %1, %ctrl_1 = ifrt.Call @f1(%0) on devices [0, 1] : (!array_m1) -> !array_m1
    // CHECK: %[[OUT2:.+]], %{{.+}} = ifrt.Call @f2(%[[OUT0]])
    %2, %ctrl_2 = ifrt.Call @f2(%0) on devices [0, 1] : (!array_m1) -> !array_m1
    return %0, %1, %2 : !array_m1, !array_m1, !array_m1
  }

  // CHECK-LABEL: func.func private @f0()
  // CHECK-NEXT:    stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT:    return
  func.func private @f0() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<8.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f1(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT:    %[[ADD1:.+]] = stablehlo.add %[[C1]], %[[C1]]
  // CHECK-NEXT:    return %[[ADD1]]
  func.func private @f1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f2(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C2:.+]] = stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT:    %[[ADD2:.+]] = stablehlo.add %[[C2]], %[[C2]]
  // CHECK-NEXT:    return %[[ADD2]]
  func.func private @f2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }
}

// -----

!array_m1 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_m2 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<1x2 to [0] on 2>, [2, 3]>

// CHECK-LABEL: module @through_reshards
module @through_reshards {
  func.func @main() -> (!array_m1, !array_m1, !array_m2)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @f0() on devices [0, 1] : () -> !array_m1
    // CHECK: ifrt.Reshard
    %t1, %ctrl_t1 = ifrt.Reshard(%0) : (!array_m1) -> !array_m2
    // CHECK: ifrt.Reshard
    %t2, %ctrl_t2 = ifrt.Reshard(%t1) : (!array_m2) -> !array_m1
    %1, %ctrl_1 = ifrt.Call @f1(%t2) on devices [0, 1]
        : (!array_m1) -> !array_m1
    %2, %ctrl_2 = ifrt.Call @f2(%t1) on devices [2, 3]
        : (!array_m2) -> !array_m2
    return %0, %1, %2 : !array_m1, !array_m1, !array_m2
  }

  // CHECK-LABEL: func.func private @f0()
  // CHECK-NEXT:    stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT:    return
  func.func private @f0() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<8.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f1(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT:    %[[ADD1:.+]] = stablehlo.add %[[C1]], %[[C1]]
  // CHECK-NEXT:    return %[[ADD1]]
  func.func private @f1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f2(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C2:.+]] = stablehlo.constant dense<8.000000e+00>
  // CHECK-NEXT:    %[[ADD2:.+]] = stablehlo.add %[[C2]], %[[C2]]
  // CHECK-NEXT:    return %[[ADD2]]
  func.func private @f2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }
}

// -----

!array_m1_2x1 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_m2_2x1 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [2, 3]>
!array_m2_1x2 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [2, 3]>
!array_m1_1x2 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @through_copy_arrays_and_reshard_chain
module @through_copy_arrays_and_reshard_chain {
  func.func @main() -> (!array_m2_2x1, !array_m1_1x2)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @f0() on devices [0, 1] : () -> !array_m1_2x1
    // CHECK: ifrt.CopyArrays
    %c1, %ctrl_c1 = ifrt.CopyArrays(%0) : (!array_m1_2x1) -> !array_m2_2x1
    // CHECK: ifrt.Reshard
    %r1, %ctrl_r1 = ifrt.Reshard(%c1) : (!array_m2_2x1) -> !array_m2_1x2
    // CHECK: ifrt.CopyArrays
    %c2, %ctrl_c2 = ifrt.CopyArrays(%r1) : (!array_m2_1x2) -> !array_m1_1x2
    %1, %ctrl_1 = ifrt.Call @f1(%c1) on devices [2, 3]
        : (!array_m2_2x1) -> !array_m2_2x1
    %2, %ctrl_2 = ifrt.Call @f2(%c2) on devices [0, 1]
        : (!array_m1_1x2) -> !array_m1_1x2
    return %1, %2 : !array_m2_2x1, !array_m1_1x2
  }

  func.func private @f0() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<4.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f1(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C1:.+]] = stablehlo.constant dense<4.000000e+00>
  // CHECK-NEXT:    %[[ADD1:.+]] = stablehlo.add %[[C1]], %[[C1]]
  // CHECK-NEXT:    return %[[ADD1]]
  func.func private @f1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f2(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C2:.+]] = stablehlo.constant dense<4.000000e+00>
  // CHECK-NEXT:    %[[MUL2:.+]] = stablehlo.multiply %[[C2]], %[[C2]]
  // CHECK-NEXT:    return %[[MUL2]]
  func.func private @f2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %mul = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf32>
    return %mul : tensor<4x8xf32>
  }
}

// -----

!array_m1_2x1 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_m2_2x1 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [2, 3]>
!array_m2_1x2 = !ifrt.array<tensor<4x8xf32>,
                            #ifrt.sharding_param<1x2 to [0] on 2>, [2, 3]>

// CHECK-LABEL: module @multi_array_reshard_and_copy_arrays
module @multi_array_reshard_and_copy_arrays {
  func.func @main(%arg0: !array_m1_2x1) -> !array_m2_2x1
      attributes {ifrt.function} {
    %0:3, %ctrl_0 = ifrt.Call @producer(%arg0) on devices [0, 1]
        : (!array_m1_2x1) -> (!array_m1_2x1, !array_m1_2x1, !array_m1_2x1)
    %r:2, %ctrl_r = ifrt.Reshard(%0#0, %0#1)
        : (!array_m1_2x1, !array_m1_2x1) -> (!array_m2_1x2, !array_m2_1x2)
    %ca:2, %ctrl_ca = ifrt.CopyArrays(%0#0, %0#2)
        : (!array_m1_2x1, !array_m1_2x1) -> (!array_m2_2x1, !array_m2_2x1)
    %1, %ctrl_1 = ifrt.Call @consumer(%ca#0, %r#1, %ca#1) on devices [2, 3]
        : (!array_m2_2x1, !array_m2_1x2, !array_m2_2x1) -> !array_m2_2x1
    return %1 : !array_m2_2x1
  }

  func.func private @producer(%arg0: tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
    %neg = stablehlo.negate %arg0 : tensor<4x8xf32>
    %c3 = stablehlo.constant dense<3.000000e+00> : tensor<4x8xf32>
    %c5 = stablehlo.constant dense<5.000000e+00> : tensor<4x8xf32>
    return %neg, %c3, %c5 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @consumer(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C3:.+]] = stablehlo.constant dense<3.000000e+00>
  // CHECK-NEXT:    %[[C5:.+]] = stablehlo.constant dense<5.000000e+00>
  // CHECK-NEXT:    %[[A:.+]] = stablehlo.add %arg0, %[[C3]]
  // CHECK-NEXT:    %[[B:.+]] = stablehlo.multiply %[[A]], %[[C5]]
  // CHECK-NEXT:    return %[[B]]
  func.func private @consumer(
      %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %a = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    %b = stablehlo.multiply %a, %arg2 : tensor<4x8xf32>
    return %b : tensor<4x8xf32>
  }
}

// -----

!array_m1 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @shared_callee_cloned_only_for_const_caller
module @shared_callee_cloned_only_for_const_caller {
  func.func @main(%arg0: !array_m1) -> (!array_m1, !array_m1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @const_producer() on devices [0, 1]
        : () -> !array_m1
    // CHECK: %[[C_OUT:.+]], %{{.+}} = ifrt.Call @[[CLONED_CALLEE:shared_add_.+]](%{{.+}})
    %1, %ctrl_1 = ifrt.Call @shared_add(%0) on devices [0, 1]
        : (!array_m1) -> !array_m1
    // CHECK: %[[NC_OUT:.+]], %{{.+}} = ifrt.Call @shared_add(%arg0)
    %2, %ctrl_2 = ifrt.Call @shared_add(%arg0) on devices [0, 1]
        : (!array_m1) -> !array_m1
    return %1, %2 : !array_m1, !array_m1
  }

  func.func private @const_producer() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<2.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @shared_add(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[ADD:.+]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT:    return %[[ADD]]
  // CHECK-LABEL: func.func private @shared_add_
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<2.000000e+00>
  // CHECK-NEXT:    %[[ADD_C:.+]] = stablehlo.add %[[C]], %[[C]]
  // CHECK-NEXT:    return %[[ADD_C]]
  func.func private @shared_add(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }
}

// -----

!array_m1 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @passthrough_call_reverse_order
module @passthrough_call_reverse_order {
  func.func @main() -> !array_m1 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @f0() on devices [0, 1] : () -> !array_m1
    %1, %ctrl_1 = ifrt.Call @f1(%0) on devices [0, 1] : (!array_m1) -> !array_m1
    %2, %ctrl_2 = ifrt.Call @f2(%1) on devices [0, 1] : (!array_m1) -> !array_m1
    return %2 : !array_m1
  }

  func.func private @f0() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<9.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f1(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<9.000000e+00>
  // CHECK-NEXT:    return %[[C]]
  func.func private @f1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @f2(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[ADD:.+]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT:    return %[[ADD]]
  func.func private @f2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }
}

// -----

!array_m1 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @external_declaration_skipped
module @external_declaration_skipped {
  func.func @main() -> (!array_m1, !array_m1) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @const_producer() on devices [0, 1]
        : () -> !array_m1
    // CHECK: ifrt.Call @external_consumer
    %1, %ctrl_1 = ifrt.Call @external_consumer(%0) on devices [0, 1]
        : (!array_m1) -> !array_m1
    %2, %ctrl_2 = ifrt.Call @external_producer() on devices [0, 1]
        : () -> !array_m1
    // CHECK: ifrt.Call @defined_consumer
    %3, %ctrl_3 = ifrt.Call @defined_consumer(%2) on devices [0, 1]
        : (!array_m1) -> !array_m1
    return %1, %3 : !array_m1, !array_m1
  }

  func.func private @const_producer() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<7.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @external_consumer(tensor<4x8xf32>) -> tensor<4x8xf32>
  func.func private @external_consumer(tensor<4x8xf32>) -> tensor<4x8xf32>

  // CHECK-LABEL: func.func private @external_producer() -> tensor<4x8xf32>
  func.func private @external_producer() -> tensor<4x8xf32>

  // CHECK-LABEL: func.func private @defined_consumer(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[ADD:.+]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT:    return %[[ADD]]
  func.func private @defined_consumer(%arg0: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  }
}

// -----

!array_m1 = !ifrt.array<tensor<4x8xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>

// CHECK-LABEL: module @multi_block_consumer
module @multi_block_consumer {
  func.func @main() -> !array_m1 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @const_producer() on devices [0, 1]
        : () -> !array_m1
    // CHECK: ifrt.Call @multi_block
    %1, %ctrl_1 = ifrt.Call @multi_block(%0) on devices [0, 1]
        : (!array_m1) -> !array_m1
    return %1 : !array_m1
  }

  func.func private @const_producer() -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<6.000000e+00> : tensor<4x8xf32>
    return %c : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func private @multi_block(%arg0: tensor<4x8xf32>)
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<6.000000e+00>
  // CHECK-NEXT:    %[[ADD:.+]] = stablehlo.add %[[C]], %[[C]]
  // CHECK-NEXT:    return %[[ADD]]
  // CHECK-NEXT:  ^bb1:
  // CHECK-NEXT:    %[[MUL:.+]] = stablehlo.multiply %[[C]], %[[C]]
  // CHECK-NEXT:    return %[[MUL]]
  func.func private @multi_block(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    return %add : tensor<4x8xf32>
  ^bb1:
    %mul = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf32>
    return %mul : tensor<4x8xf32>
  }
}

