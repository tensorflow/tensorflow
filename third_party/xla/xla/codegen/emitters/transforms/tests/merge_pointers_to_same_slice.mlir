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
// RUN: emitters_opt %s -split-input-file -xla-lower-tensors -xla-merge-pointers | FileCheck %s

module {
  func.func private @tensorargs(%arg0: tensor<43xf32> {xla.slice_index = 0},
                        %arg1: tensor<43xf32> {xla.slice_index = 1, xla.invariant},
                        %arg2: tensor<43xf32> {xla.slice_index = 0},
                        %arg3: index) -> f32 {
    %v0 = tensor.extract %arg0[%arg3] : tensor<43xf32>
    %v1 = tensor.extract %arg1[%arg3] : tensor<43xf32>
    %v2 = tensor.extract %arg2[%arg3] : tensor<43xf32>
    %sum = arith.addf %v0, %v1 : f32
    %sum2 = arith.addf %sum, %v2 : f32
    func.return %sum2 : f32
  }

  func.func @tensorcall(%arg0: tensor<43xf32> {xla.slice_index = 0},
                        %arg1: tensor<43xf32> {xla.slice_index = 1, xla.invariant},
                        %arg2: tensor<43xf32> {xla.slice_index = 0},
                        %arg3: index) -> f32 attributes {xla.entry} {
    %call = func.call @tensorargs(%arg0, %arg1, %arg2, %arg3) :
      (tensor<43xf32>, tensor<43xf32>, tensor<43xf32>, index) -> f32
    func.return %call : f32
  }
}

// CHECK:      func.func private @tensorargs(
// CHECK-SAME:   %[[ARG0:.*]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:   %[[ARG1:.*]]: !llvm.ptr {llvm.noalias, llvm.readonly, xla.invariant},
// CHECK-SAME:   %[[ARG2:.*]]: index) -> f32 {
// CHECK:        %[[GEP0:.*]] = llvm.getelementptr inbounds %[[ARG0]]
// CHECK:        llvm.load %[[GEP0]] : !llvm.ptr
// CHECK:        %[[GEP1:.*]] = llvm.getelementptr inbounds %[[ARG1]]
// CHECK:        llvm.load %[[GEP1]] : !llvm.ptr
// CHECK:        %[[GEP2:.*]] = llvm.getelementptr inbounds %[[ARG0]]

// CHECK:      func.func @tensorcall
// CHECK-SAME:   %[[ARG0:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: !llvm.ptr {llvm.noalias, llvm.readonly, xla.invariant},
// CHECK-SAME:   %[[ARG2:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:   %[[ARG3:[a-z0-9]+]]: index) -> f32 attributes {xla.entry} {
// CHECK:        call @tensorargs(%[[ARG0]], %[[ARG1]], %[[ARG3]])

// -----

func.func @wrapping_func(%arg0: tensor<43xf32>,
                         %arg1: tensor<43xf32>,
                         %idx: index) -> f32 {
  %result = func.call @private_entry_func(%arg0, %arg1, %idx) :
      (tensor<43xf32>, tensor<43xf32>, index) -> f32
  func.return %result : f32
}

func.func private @private_entry_func(
  %arg0: tensor<43xf32> {xla.slice_index = 0},
  %arg1: tensor<43xf32> {xla.slice_index = 0},
  %idx: index) -> f32 attributes {xla.entry} {
  %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
  %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
  %sum = arith.addf %v0, %v1 : f32
  func.return %sum : f32
}

// CHECK-LABEL: func.func @wrapping_func(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 {
// CHECK:         %[[RESULT:.*]] = call @private_entry_func(
// CHECK-SAME:      %[[ARG0]], %[[ARG1]], %[[IDX]])
// CHECK:         return %[[RESULT]] : f32
// CHECK:       }

// CHECK-LABEL: func.func private @private_entry_func(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry} {
// CHECK:         %[[GEP0:.*]] = llvm.getelementptr inbounds %[[ARG0]]
// CHECK:         %[[V0:.*]] = llvm.load %[[GEP0]]
// CHECK:         %[[GEP1:.*]] = llvm.getelementptr inbounds %[[ARG0]]
// CHECK:         %[[V1:.*]] = llvm.load %[[GEP1]]
// CHECK:         %[[SUM:.*]] = arith.addf %[[V0]], %[[V1]]
// CHECK:         return %[[SUM]]
// CHECK:        }

// -----

// Verify that xla.aliased suppresses llvm.noalias on that argument.

func.func @aliased_suppresses_noalias(
    %arg0: tensor<43xf32> {xla.slice_index = 0, xla.aliased},
    %arg1: tensor<43xf32> {xla.slice_index = 1},
    %arg2: tensor<43xf32> {xla.slice_index = 2, xla.invariant},
    %idx: index) -> f32 attributes {xla.entry} {
  %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
  %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
  %v2 = tensor.extract %arg2[%idx] : tensor<43xf32>
  %sum = arith.addf %v0, %v1 : f32
  %sum2 = arith.addf %sum, %v2 : f32
  func.return %sum2 : f32
}

// CHECK-LABEL: func.func @aliased_suppresses_noalias(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[ARG2:[a-z0-9]+]]: !llvm.ptr {llvm.noalias, llvm.readonly, xla.invariant},
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry}
// CHECK-NOT:     xla.aliased

// -----

// Merged arguments keep only what all of them promise: an aliased non invariant
// use of the slice leaves the survivor neither readonly nor noalias.

func.func @mixed_invariance(
    %arg0: tensor<43xf32> {xla.slice_index = 0, xla.invariant},
    %arg1: tensor<43xf32> {xla.slice_index = 0, xla.aliased},
    %idx: index) -> f32 attributes {xla.entry} {
  %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
  %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
  %sum = arith.addf %v0, %v1 : f32
  func.return %sum : f32
}

// CHECK-LABEL: func.func @mixed_invariance(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry}
// CHECK:         llvm.getelementptr inbounds %[[ARG0]]
// CHECK:         llvm.getelementptr inbounds %[[ARG0]]

// -----

// Kept duplicates of an entry function get the same pointer, so each one is
// weakened too, regardless of argument order.

func.func @mixed_invariance_reversed(
    %arg0: tensor<43xf32> {xla.slice_index = 0, xla.aliased},
    %arg1: tensor<43xf32> {xla.slice_index = 0, xla.invariant},
    %idx: index) -> f32 attributes {xla.entry} {
  %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
  %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
  %sum = arith.addf %v0, %v1 : f32
  func.return %sum : f32
}

// CHECK-LABEL: func.func @mixed_invariance_reversed(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry}
// -----

// Merging an invariant argument with a written slice drops readonly while
// keeping noalias on both entry arguments.

func.func @invariant_and_written(
    %arg0: tensor<43xf32> {xla.slice_index = 0, xla.invariant},
    %arg1: tensor<43xf32> {xla.slice_index = 0},
    %idx: index) -> f32 attributes {xla.entry} {
  %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
  %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
  %sum = arith.addf %v0, %v1 : f32
  func.return %sum : f32
}

// CHECK-LABEL: func.func @invariant_and_written(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry}

// -----

// On a private function, duplicate slice arguments are erased and the survivor
// keeps only the promises shared across its duplicates.

module {
  func.func private @private_mixed(
      %arg0: tensor<43xf32> {xla.slice_index = 0, xla.invariant},
      %arg1: tensor<43xf32> {xla.slice_index = 0},
      %arg2: tensor<43xf32> {xla.slice_index = 1, xla.invariant},
      %arg3: tensor<43xf32> {xla.slice_index = 1, xla.aliased},
      %idx: index) -> f32 {
    %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
    %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
    %v2 = tensor.extract %arg2[%idx] : tensor<43xf32>
    %v3 = tensor.extract %arg3[%idx] : tensor<43xf32>
    %s0 = arith.addf %v0, %v1 : f32
    %s1 = arith.addf %v2, %v3 : f32
    %s2 = arith.addf %s0, %s1 : f32
    func.return %s2 : f32
  }

  func.func @call_private_mixed(
      %arg0: tensor<43xf32> {xla.slice_index = 0, xla.invariant},
      %arg1: tensor<43xf32> {xla.slice_index = 0},
      %arg2: tensor<43xf32> {xla.slice_index = 1, xla.invariant},
      %arg3: tensor<43xf32> {xla.slice_index = 1, xla.aliased},
      %idx: index) -> f32 attributes {xla.entry} {
    %call = func.call @private_mixed(%arg0, %arg1, %arg2, %arg3, %idx) :
        (tensor<43xf32>, tensor<43xf32>, tensor<43xf32>, tensor<43xf32>, index) -> f32
    func.return %call : f32
  }
}

// CHECK-LABEL: func.func private @private_mixed(
// CHECK-SAME:    %[[P0:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[P1:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[PIDX:[a-z0-9]+]]: index) -> f32
// CHECK:       func.func @call_private_mixed(
// CHECK-SAME:    %[[A0:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[A1:[a-z0-9]+]]: !llvm.ptr {llvm.noalias},
// CHECK-SAME:    %[[A2:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[A3:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[AIDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry}
// CHECK:         call @private_mixed(%[[A0]], %[[A2]], %[[AIDX]])

// -----

// xla.aliased suppresses llvm.noalias while keeping llvm.readonly on an
// invariant argument, and propagates across merged duplicate slice arguments.

func.func @aliased_readonly(
    %arg0: tensor<43xf32> {xla.slice_index = 0, xla.invariant},
    %arg1: tensor<43xf32> {xla.slice_index = 0, xla.invariant, xla.aliased},
    %arg2: tensor<43xf32> {xla.slice_index = 1, xla.invariant},
    %idx: index) -> f32 attributes {xla.entry} {
  %v0 = tensor.extract %arg0[%idx] : tensor<43xf32>
  %v1 = tensor.extract %arg1[%idx] : tensor<43xf32>
  %v2 = tensor.extract %arg2[%idx] : tensor<43xf32>
  %s0 = arith.addf %v0, %v1 : f32
  %s1 = arith.addf %s0, %v2 : f32
  func.return %s1 : f32
}

// CHECK-LABEL: func.func @aliased_readonly(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: !llvm.ptr {llvm.readonly, xla.invariant},
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: !llvm.ptr {llvm.readonly, xla.invariant},
// CHECK-SAME:    %[[ARG2:[a-z0-9]+]]: !llvm.ptr {llvm.noalias, llvm.readonly, xla.invariant},
// CHECK-SAME:    %[[IDX:[a-z0-9]+]]: index) -> f32 attributes {xla.entry}
// CHECK-NOT:     xla.aliased
