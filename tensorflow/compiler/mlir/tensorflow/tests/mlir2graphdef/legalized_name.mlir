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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  tf_executor.graph {
    // CHECK: name: ".foo"
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>) loc("^foo")
    // CHECK: name: "fo.o"
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> (tensor<i32>) loc("fo{o")
    // CHECK: name: "foo"
    %2:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("foo@1")
    // CHECK: name: "ba.r"
    %3:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("ba r")
    // CHECK: name: "2"
    %4:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("2")
    // CHECK: name: "_3"
    %5:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("_3")
    // CHECK: name: "foo_"
    %6:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("foo_")
    tf_executor.fetch
  }
  func.return
}
