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
// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_output_shapes(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2xi32> {tfg.regenerate_output_shapes},
// CHECK-NEXT: %[[ARG1:.*]]: tensor<*xi32>)
// CHECK-NEXT: -> (tensor<2xi32>)
tfg.func @test_output_shapes(%arg0: tensor<*xi32> {tf._output_shapes = [#tf_type.shape<2>]},
                             %arg1: tensor<*xi32>)
    -> (tensor<*xi32>) {
  // CHECK: return(%[[ARG0]]) : tensor<2xi32>
  return(%arg0) : tensor<*xi32>
}

// -----

// Check that output shapes that is not an array is ignored.

// CHECK-LABEL: tfg.func @test_not_array_attr(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xi32> {tf._output_shapes = 5 : i32})
tfg.func @test_not_array_attr(%arg0: tensor<*xi32> {tf._output_shapes = 5 : i32})
    -> (tensor<*xi32>) {
  return(%arg0) : tensor<*xi32>
}

// -----

// Check that output shapes that is not an array of shapes is ignored.

// CHECK-LABEL: tfg.func @test_not_shape_arr(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xi32> {tf._output_shapes = [5 : i32]})
tfg.func @test_not_shape_arr(%arg0: tensor<*xi32> {tf._output_shapes = [5 : i32]})
    -> (tensor<*xi32>) {
  return(%arg0) : tensor<*xi32>
}

// -----

// Check that output shapes that is an array of shapes but has the wrong number
// of shapes is ignored.

// CHECK-LABEL: tfg.func @test_wrong_shape_list_size(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xi32> {tf._output_shapes = [{{.*}}]})
tfg.func @test_wrong_shape_list_size(%arg0: tensor<*xi32> {tf._output_shapes = [
  #tf_type.shape<2>, #tf_type.shape<2>
]}) -> (tensor<*xi32>) {
  return(%arg0) : tensor<*xi32>
}
