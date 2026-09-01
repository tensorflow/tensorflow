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
    // CHECK: %[[VAR:.*]], {{.*}} name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = i32, shape = #tf_type.shape<2x3>, shared_name = ""} : () -> (tensor<2x3x!tf_type.int32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = i32, shape = #tf_type.shape<1x2x3x1>, shared_name = ""} : () -> (tensor<1x2x3x1x!tf_type.int32ref>)
    // CHECK: Identity(%[[VAR]]) name("s1")
    %Squeeze, %ctl_2 = Squeeze(%VariableV2) name("s1") {T = i32, squeeze_dims = []} : (tensor<2x3x!tf_type.int32ref>) -> (tensor<*xi32>)
    %Squeeze_3, %ctl_4 = Squeeze(%VariableV2_0) name("s2") {T = i32, squeeze_dims = []} : (tensor<1x2x3x1x!tf_type.int32ref>) -> (tensor<*xi32>)
    %Add, %ctl_5 = Add(%Squeeze, %Squeeze_3) name("out") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    return
  }
}
