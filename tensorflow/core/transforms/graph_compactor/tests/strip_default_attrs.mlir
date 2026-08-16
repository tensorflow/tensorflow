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
// RUN: tfg-transforms-opt --tfg-strip-default-attrs %s | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: VarHandleOp
  // CHECK-NOT: container
  // CHECK-NOT: shared_name
  // CHECK-NOT: allowed_devices
  // CHECK: dtype
  // CHECK: shape
  %Var, %ctl = VarHandleOp {
    container = "", shared_name = "", dtype = i32, 
    shape = #tf_type.shape<4x4>, allowed_devices = []
  } : () -> (tensor<*x!tf_type.resource>)
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: VarHandleOp
  // CHECK: container
  // CHECK-NOT: allowed_devices
  // CHECK: dtype
  // CHECK: shape
  // CHECK: shared_name
  %Var, %ctl = VarHandleOp {
    container = "foo", shared_name = "bar", dtype = i32, 
    shape = #tf_type.shape<4x4>, allowed_devices = []
  } : () -> (tensor<*x!tf_type.resource>)
}