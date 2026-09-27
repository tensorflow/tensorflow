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
// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @body
tfg.func @body(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
  %A, %ctl = A(%arg0, %arg1) : (tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%A) : tensor<*xf32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[START:.*]], %[[CTL:.*]] = Start
  %Start, %ctl = Start : () -> (tensor<i32>)
  // CHECK-NEXT: %[[LIMIT:.*]], %[[CTL_0:.*]] = Limit
  %Limit, %ctl_0 = Limit : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DELTA:.*]], %[[CTL_1:.*]] = Delta
  %Delta, %ctl_1 = Delta : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DATA:.*]], %[[CTL_2:.*]] = Data
  %Data, %ctl_2 = Data : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[FOR:.*]], %[[CTL_5:.*]] = ForRegion(%[[DATA]]) from %[[START]] to %[[LIMIT]] by %[[DELTA]]  {
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<{{.*}}>,
  // CHECK-SAME:      %[[ARG2:.*]]: !tf_type.control, %[[ARG3:.*]]: !tf_type.control):
  // CHECK-NEXT:   %[[A:.*]], %[[CTL_6:.*]] = A(%[[ARG0]], %[[ARG1]])
  // CHECK-NEXT:   yield(%[[A]])
  // CHECK-NEXT: } {_some_attr, body_attrs = {}
  %For, %ctl_3 = For(%Start, %Limit, %Delta, %Data)
                 {T = [f32], _some_attr, body = #tf_type.func<@body, {}>}
                 : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: Sink [%[[CTL_5]]]
  %ctl_4 = Sink [%ctl_3]
}
