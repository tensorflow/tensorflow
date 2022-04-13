// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-quant-opt %s -quant-add-main-function -allow-unregistered-dialect -mlir-disable-threading | FileCheck %s

// CHECK-LABEL: module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    func.return
  }
// CHECK: func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]}

  func.func @mul1(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["mul1"]} {
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }
// CHECK: func private @mul1(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0", outputs = "PartitionedCall:0"}}
// CHECK:   %[[MUL_0:.*]] = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   return %[[MUL_0]] : tensor<1xf32>
// CHECK: }

  func.func @mul2(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "mul2_y:0,mul2_x:0", outputs = "PartitionedCall_1:0"}, tf_saved_model.exported_names = ["mul2"]} {
    %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = "tf.Mul"(%0, %cst) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
    func.return %1 : tensor<1xf32>
  }
// CHECK: func private @mul2(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {tf.entry_function = {inputs = "mul2_y:0,mul2_x:0", outputs = "PartitionedCall_1:0"}} {
// CHECK:   %[[CONST_0:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK:   %[[MUL_1:.*]] = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   %[[MUL_2:.*]] = "tf.Mul"(%[[MUL_1]], %[[CONST_0]]) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK:   return %[[MUL_2]] : tensor<1xf32>
// CHECK: }

// CHECK: func @main(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["mul1_y:0"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["mul1_x:0"]}, %arg2: tensor<1xf32> {tf_saved_model.index_path = ["mul2_y:0"]}, %arg3: tensor<1xf32> {tf_saved_model.index_path = ["mul2_x:0"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall:0"]}, tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall_1:0"]}) attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0,mul2_y:0,mul2_x:0", outputs = "PartitionedCall:0,PartitionedCall_1:0"}, tf_saved_model.exported_names = ["main"]} {
// CHECK-NOT: f = @NoOp
// CHECK:   %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @mul1} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   %[[PARTITIONEDCALL_1:.*]] = "tf.PartitionedCall"(%arg2, %arg3) {config = "", config_proto = "", executor_type = "", f = @mul2} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   return %[[PARTITIONEDCALL_0]], %[[PARTITIONEDCALL_1]] : tensor<1xf32>, tensor<1xf32>
// CHECK: }
}
