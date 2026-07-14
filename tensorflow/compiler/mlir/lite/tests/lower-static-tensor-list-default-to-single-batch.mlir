// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: litert-opt "-tfl-lower-static-tensor-list=allow-tensorlist-pass-through default-to-single-batch=false" -split-input-file %s | FileCheck %s

// -----

func.func @tensorlistReserveConstantUnknownElementShapeDim(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<?x7xf32> {
  %cst = arith.constant dense<[-1, 7]> : tensor<2xi32>
  %0 = "tf.TensorListReserve"(%cst, %arg0) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x7xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg1, %cst) : (tensor<!tf_type.variant<tensor<?x7xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<?x7xf32>
  func.return %1 : tensor<?x7xf32>

// CHECK-LABEL: tensorlistReserveConstantUnknownElementShapeDim
// CHECK:      "tf.TensorListReserve"
// CHECK:      "tf.TensorListGetItem"
}
