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
// RUN: tfg-transforms-opt --tfg-lift-legacy-call --symbol-privatize --symbol-dce %s | FileCheck %s

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: Foo {tfg.legacy_call = @Foo}
  %Foo, %ctl = Foo : () -> (tensor<i1>)
}

// CHECK: tfg.func private @Foo
tfg.func private @Foo() -> (tensor<i1>) {
  // CHECK-NOT: tfg.legacy_call
  %Const, %ctl = Const {dtype = i1, value = dense<0> : tensor<i1>} : () -> (tensor<i1>)
  return(%Const) : tensor<i1>
}

// CHECK-NOT: tfg.func @Bar
tfg.func private @Bar() -> (tensor<i1>) {
  %Const, %ctl = Const {dtype = i1, value = dense<0> : tensor<i1>} : () -> (tensor<i1>)
  return(%Const) : tensor<i1>
}