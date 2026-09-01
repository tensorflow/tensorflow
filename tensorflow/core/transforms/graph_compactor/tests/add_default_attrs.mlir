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
// RUN: tfg-transforms-opt --tfg-add-default-attrs %s | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: VarHandleOp
  // CHECK: allowed_devices = []
  // CHECK: container = ""
  // CHECK: shared_name = ""
  %Var, %ctl = VarHandleOp {
    dtype = i32, shape = #tf_type.shape<4x4> 
  } : () -> (tensor<*x!tf_type.resource>)
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: VarHandleOp
  // CHECK: allowed_devices = []
  // CHECK: container = "foo"
  // CHECK: shared_name = "bar"
  %Var, %ctl = VarHandleOp {
    container = "foo", shared_name = "bar",
    dtype = i32, shape = #tf_type.shape<4x4>
  } : () -> (tensor<*x!tf_type.resource>)
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // Unregistered op
  // CHECK: FooUnregisteredOp
  // CHECK-SAME: {a}
  %FooUnregisteredOp, %ctl = FooUnregisteredOp {a} : () -> (tensor<i32>)
}

// CHECK-LABEL: tfg.func @op_no_default_attrs
tfg.func @op_no_default_attrs(%lhs: tensor<*xi32>, %rhs: tensor<*xi32>) -> (tensor<*xi32>) {
  // Op with no default-valued attributes
  // CHECK: Add
  // CHECK-SAME: {T = i32}
  %Add, %ctl = Add(%lhs, %rhs) {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  return(%Add) : tensor<*xi32>
}
