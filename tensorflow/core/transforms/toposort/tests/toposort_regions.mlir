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
// RUN: tfg-transforms-opt %s --pass-pipeline="builtin.module(tfg.graph(tfg-toposort), tfg.func(tfg-toposort))" | FileCheck %s

// Test with region ops
// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK-NEXT: %[[INDEX:.*]], %{{.*}} = Index
  // CHECK-NEXT: %[[FOO:.*]], %{{.*}} = Foo(%[[INDEX]]) :
  // CHECK-NEXT: %{{.*}}, %[[CTLCASE:.*]] = CaseRegion %[[INDEX]] {
  // CHECK-NEXT:   %[[B:.*]], %{{.*}} = B(%[[A:.*]]) :
  // CHECK-NEXT:   %[[A]], %{{.*}} = A(%[[FOO]]) :
  // CHECK-NEXT:   yield(%[[B]])
  // CHECK-NEXT: }
  // CHECK-NEXT: NoOp [%[[CTLCASE]]]
  %ctlNoOp = NoOp [%ctlCase]
  %Case, %ctlCase = CaseRegion %Index {
    %B, %ctlB = B(%A) : (tensor<*xi32>) -> (tensor<*xi32>)
    %A, %ctlA = A(%Foo) : (tensor<*xi32>) -> (tensor<*xi32>)
    yield(%B) : tensor<*xi32>
  } : (tensor<*xi32>) -> (tensor<*xi32>)
  %Foo, %ctlFoo = Foo(%Index) : (tensor<*xi32>) -> (tensor<*xi32>)
  %Index, %ctlIndex = Index : () -> (tensor<*xi32>)
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK-NEXT: %[[A:.*]], %{{.*}} = Placeholder name("a")
  // CHECK-NEXT: %[[COND:.*]], %{{.*}} = Placeholder name("cond")
  // CHECK-NEXT: IfRegion %[[COND]] then {
  // CHECK-NEXT:   %[[C:.*]], %{{.*}} = Placeholder name("c")
  // CHECK-NEXT:   %[[D:.*]], %{{.*}} = Foo(%[[C]]) name("d")
  // CHECK-NEXT:   yield(%[[D]])
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   yield(%[[A]])
  %a, %ctlA = Placeholder name("a") : () -> (tensor<*xi32>)
  %cond, %ctlCond = Placeholder name("cond") : () -> (tensor<*xi1>)
  %b, %ctlB = IfRegion %cond then {
    %c, %ctlC = Placeholder name("c") : () -> (tensor<*xi32>)
    %d, %ctlD = Foo(%c) name("d") : (tensor<*xi32>) -> (tensor<*xi32>)
    yield(%d) : tensor<*xi32>
  } else {
    yield(%a) : tensor<*xi32>
  } : (tensor<*xi1>) -> (tensor<*xi32>)
}
