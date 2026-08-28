// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt -allow-unregistered-dialect -tf-materialize-passthrough-op %s  | FileCheck %s


// Check that the MlirPassthroughOp is eliminated and replaced by its attached
// MLIR module.

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<10xf32>)
// CHECK-NEXT:    %[[ADD:.*]] = "tf.Add"(%[[ARG0]], %[[ARG1]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[MAGIC:.*]] = "magic.op"(%[[ADD]], %[[ADD]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    return %[[MAGIC]]
  %0 = "tf.MlirPassthroughOp"(%arg0, %arg1) {Tinputs = ["tfdtype$DT_FLOAT", "tfdtype$DT_FLOAT"], Toutputs = ["tfdtype$DT_FLOAT"], device = "", mlir_module = "\0Afunc.func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {\0A   %add = \22tf.Add\22(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>\0A   %ret = \22magic.op\22(%add, %add) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>\0A   func.return %ret : tensor<10x10xf32>\0A}\0A", name = "MlirPassthroughOp"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
}
