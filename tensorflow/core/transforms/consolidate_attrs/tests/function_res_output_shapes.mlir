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

// CHECK-LABEL: tfg.func @test_output_shapes
// CHECK: -> (tensor<4xi32> {tfg.regenerate_output_shapes})
tfg.func @test_output_shapes(%arg0: tensor<*xi32>)
    -> (tensor<*xi32> {tf._output_shapes = [#tf_type.shape<4>]}) {
  // CHECK: %[[A:.*]], %{{.*}} = A(%{{.*}}) : (tensor<*xi32>) -> (tensor<4xi32>)
  %A, %ctl = A(%arg0) : (tensor<*xi32>) -> (tensor<*xi32>)
  // CHECK: return(%[[A]]) : tensor<4xi32>
  return(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_output_shapes_arg_type(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xi32>)
// CHECK-NEXT: -> (tensor<4xi32> {tfg.regenerate_output_shapes})
tfg.func @test_output_shapes_arg_type(%arg0: tensor<*xi32>)
    -> (tensor<*xi32> {tf._output_shapes = [#tf_type.shape<4>]}) {
  // CHECK: return(%[[ARG0]]) : tensor<4xi32>
  return(%arg0) : tensor<*xi32>
}
