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
// RUN: tfg-transforms-opt -pass-pipeline='builtin.module(tfg.func(tfg-name-compress))' %s | FileCheck %s

// CHECK-LABEL: tfg.func @foo
// CHECK-SAME: tfg.name = "A"
// CHECK-NEXT: tfg.name = "B"
// CHECK-NEXT: -> 
// CHECK-SAME: tfg.name = "C"
// CHECK-NEXT: tfg.name = "D"
tfg.func @foo(%argument0: tensor<i1> {tfg.name = "argument0"},
              %argument1: tensor<i1> {tfg.name = "argument1"})
    -> (tensor<i1> {tfg.name = "result0"},
        tensor<i1> {tfg.name = "result1"}) {
  // CHECK: A({{.*}}) name("F")
  %A, %ctlA = A(%argument0) name("operation0") : (tensor<i1>) -> (tensor<i1>)
  // CHECK-NEXT: B({{.*}}) name("G")
  %B, %ctlB = B(%argument1) name("operation1") : (tensor<i1>) -> (tensor<i1>)
  // CHECK-NEXT: NoOp [{{.*}}] name("H")
  %ctlC = NoOp [%ctlA, %ctlB] name("operation2")
  // CHECK-NEXT: {tfg.name = "E"}
  return(%A, %B) [%ctlC {tfg.name = "control_result0"}] : tensor<i1>, tensor<i1>
}
