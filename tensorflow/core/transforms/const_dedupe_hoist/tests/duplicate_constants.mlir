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
// RUN: tfg-transforms-opt --tfg-dedupe-hoist-constant %s | FileCheck %s

// CHECK-LABEL:   tfg.graph
tfg.graph #tf_type.version<producer = 1015, min_consumer = 0> {
  %Const, %ctl = Const device("/job:host/task:0/device:CPU:0") name("apple") {dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %Const_1, %ctl_1 = Const [%ctl] device("/job:host/task:0/device:CPU:0") name("banana") {dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %Const_2, %ctl_2 = Const [%ctl_1] device("/job:host/task:0/device:CPU:1") name("pear") {dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
}

// CHECK:   %[[VAL_0:.*]], %[[VAL_1:.*]] = Const device("/job:host/task:0/device:CPU:0") name("apple")
// CHECK:   %[[VAL_2:.*]], %[[VAL_3:.*]] = Const {{\[}}%[[VAL_1]]] device("/job:host/task:0/device:CPU:1") name("pear")
