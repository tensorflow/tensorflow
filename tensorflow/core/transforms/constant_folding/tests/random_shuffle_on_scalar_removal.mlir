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
// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<>, shared_name = ""} : () -> (tensor<!tf_type.f32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<>, shared_name = ""} : () -> (tensor<!tf_type.f32ref>)
    // CHECK: %[[R1:.*]], {{.*}} Identity{{.*}} name("s1")
    %RandomShuffle, %ctl_2 = RandomShuffle(%VariableV2) name("s1") {T = f32, seed = 0 : i64, seed2 = 0 : i64} : (tensor<!tf_type.f32ref>) -> (tensor<*xf32>)
    // CHECK: %[[R2:.*]], {{.*}} Identity{{.*}} name("s2")
    %RandomShuffle_3, %ctl_4 = RandomShuffle(%VariableV2_0) [%ctl] name("s2") {T = f32, seed = 0 : i64, seed2 = 0 : i64} : (tensor<!tf_type.f32ref>) -> (tensor<*xf32>)
    // CHECK: Add(%[[R1]], %[[R2]]) name("out1")
    %Add, %ctl_5 = Add(%RandomShuffle, %RandomShuffle_3) name("out1") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Identity, %ctl_6 = Identity(%RandomShuffle_3) name("out2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
