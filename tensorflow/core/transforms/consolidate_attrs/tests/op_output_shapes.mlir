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

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
}

// -----

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: %[[A:.*]], %{{.*}} = A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
  // CHECK: Sink(%[[A]]) : tensor<4xi32>
  %ctl_0 = Sink(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_result_type
// CHECK-NEXT: -> (tensor<4xi32>)
tfg.func @test_result_type(%arg0: tensor<i32>) -> (tensor<*xi32>) {
  // CHECK: %[[A:.*]], %{{.*}} = A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
  // CHECK: return(%[[A]]) : tensor<4xi32>
  return(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_ignore_invalid_shape
tfg.func @test_ignore_invalid_shape(%arg0: tensor<*xi32>) -> (tensor<*xi32>) {
  // CHECK: %[[A:.*]], %{{.*}} = A {_output_shapes = []} : () -> (tensor<*xi32>)
  %A, %ctl = A {_output_shapes = []} : () -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}
