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
// RUN: mlir-hlo-opt -split-input-file -lower-index-cast %s | FileCheck %s

// index_cast of static tensor
// CHECK-LABEL: func @f
func.func @f(%arg0 : tensor<10xi32>) -> tensor<10xindex> {
  // CHECK: %[[TENSOR:.*]] = tensor.generate {
  // CHECK: ^bb0(%arg1: index):
  // CHECK:   %[[E:.*]] = tensor.extract %arg0[%arg1] : tensor<10xi32>
  // CHECK:   %[[C:.*]] = arith.index_cast %[[E]] : i32 to index
  // CHECK:   tensor.yield %[[C]] : index
  // CHECK: } : tensor<10xindex>
  // CHECK: return %[[TENSOR]] : tensor<10xindex>
  %0 = arith.index_cast %arg0 : tensor<10xi32> to tensor<10xindex>
  func.return %0 : tensor<10xindex>
}

// -----

// index_cast of dynamic tensor
func.func @f(%arg0 : tensor<?xi32>) -> tensor<?xindex> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?xi32>
  // CHECK: %[[TENSOR:.*]] = tensor.generate %[[DIM]] {
  // CHECK: ^bb0(%arg1: index):
  // CHECK:   %[[E:.*]] = tensor.extract %arg0[%arg1] : tensor<?xi32>
  // CHECK:   %[[C:.*]] = arith.index_cast %[[E]] : i32 to index
  // CHECK:   tensor.yield %[[C]] : index
  // CHECK: } : tensor<?xindex>
  // CHECK: return %[[TENSOR]] : tensor<?xindex>
  %0 = arith.index_cast %arg0 : tensor<?xi32> to tensor<?xindex>
  func.return %0 : tensor<?xindex>
}

// -----

// index_cast of dynamic multidimensional tensor
func.func @f(%arg0 : tensor<42x?xi32>) -> tensor<42x?xindex> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %arg0, %[[C1]] : tensor<42x?xi32>
  // CHECK: %[[TENSOR:.*]] = tensor.generate %[[DIM]] {
  // CHECK: ^bb0(%arg1: index, %arg2: index):
  // CHECK:   %[[E:.*]] = tensor.extract %arg0[%arg1, %arg2] : tensor<42x?xi32>
  // CHECK:   %[[C:.*]] = arith.index_cast %[[E]] : i32 to index
  // CHECK:   tensor.yield %[[C]] : index
  // CHECK: } : tensor<42x?xindex>
  // CHECK: return %[[TENSOR]] : tensor<42x?xindex>
  %0 = arith.index_cast %arg0 : tensor<42x?xi32> to tensor<42x?xindex>
  func.return %0 : tensor<42x?xindex>
}

// -----

// CHECK-LABEL: func @index_castui
func.func @index_castui(%arg0 : tensor<10xi32>) -> tensor<10xindex> {
  // CHECK: tensor.generate {
  // CHECK:   %[[C:.*]] = arith.index_castui
  // CHECK:   tensor.yield %[[C]] : index
  %0 = arith.index_castui %arg0 : tensor<10xi32> to tensor<10xindex>
  func.return %0 : tensor<10xindex>
}
