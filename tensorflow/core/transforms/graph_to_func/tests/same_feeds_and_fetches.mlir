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
// RUN: tfg-transforms-opt -pass-pipeline='builtin.module(tfg-lift-graph-to-func{feeds=Placeholder1 fetches=Placeholder1})' %s | FileCheck %s

// CHECK:   tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder1", 0 : index], tfg.name = "Placeholder1_0"}
// CHECK-NEXT: -> (tensor<*xf32> {tfg.name = "Placeholder1_0"})
// CHECK:   tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>
tfg.graph #tf_type.version<producer = 34, min_consumer = 5> {
  %Placeholder, %ctl_0 = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[PLACEHOLDER1:.*]], {{.*}} Placeholder name("Placeholder2")
  %Placeholder_1, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[ADD1:.*]], {{.*}} = Add(%Placeholder1_0, %[[PLACEHOLDER1]]) name("SomeAdd1")
  %add1, %ctl2 = Add(%Placeholder, %Placeholder_1) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: Add(%[[ADD1]], %[[PLACEHOLDER1]]) name("SomeAdd2")
  %add2, %ctl3 = Add(%add1, %Placeholder_1) name("SomeAdd2") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: return(%Placeholder1_0)
}
