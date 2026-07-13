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
// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

tfg.graph #tf_type.version<producer = 0, min_consumer = 0> {
// CHECK: name: "float"
// CHECK: float_val: -0
  %Const, %ctl = Const name("float") {dtype = f32, value = dense<-0.000000e+00> : tensor<f32>} : () -> (tensor<f32>)
// CHECK: name: "half"
// CHECK: half_val: 32768
  %Const_0, %ctl_1 = Const name("half") {dtype = f16, value = dense<-0.000000e+00> : tensor<f16>} : () -> (tensor<f16>)
// CHECK: name: "complex"
// CHECK: scomplex_val: -0
// CHECK: scomplex_val: -0
  %Const_2, %ctl_3 = Const name("complex") {dtype = complex<f32>, value = dense<(-0.000000e+00,-0.000000e+00)> : tensor<complex<f32>>} : () -> (tensor<complex<f32>>)
}
