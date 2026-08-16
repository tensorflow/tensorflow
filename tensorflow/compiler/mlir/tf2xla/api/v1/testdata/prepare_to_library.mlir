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

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
    %0 = "tf.Const"() <{value = dense<-1> : tensor<3360x8xi32>}> : () -> tensor<3360x8xi32>
    %cst_33 = "tf.Const"() <{value = dense<[1120, -1]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %cst_34 = "tf.Const"() <{value = dense<[3, 1120, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %cst_63 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %1965:4 = "tf._XlaHostComputeMlir"(%0, %cst_34, %cst_63, %cst_33) <{host_mlir_module = "#loc1 = loc(\22Reshape:\22)\0A#loc2 = loc(\22Reshape_4\22)\0A#loc3 = loc(\22Reshape\22)\0A#loc9 = loc(fused[#loc1, #loc2, #loc3])\0Amodule {\0A  func.func @host_func(%arg0: tensor<3360x?xi32> loc(fused[#loc1, #loc2, #loc3]), %arg1: tensor<3xi32> loc(fused[#loc1, #loc2, #loc3]), %arg2: tensor<i32> loc(fused[#loc1, #loc2, #loc3]), %arg3: tensor<2xi32> loc(fused[#loc1, #loc2, #loc3])) -> (tensor<1x1120x?xi32>, tensor<1x1120x?xi32>, tensor<1120x?xi32>, tensor<2xi32>) {\0A    %0 = \22tf.Reshape\22(%arg0, %arg1) {_xla_outside_compilation = \220\22} : (tensor<3360x?xi32>, tensor<3xi32>) -> tensor<3x1120x?xi32> loc(#loc9)\0A    %1:3 = \22tf.Split\22(%arg2, %0) {_xla_outside_compilation = \220\22} : (tensor<i32>, tensor<3x1120x?xi32>) -> (tensor<1x1120x?xi32>, tensor<1x1120x?xi32>, tensor<1x1120x?xi32>) loc(#loc10)\0A    %2 = \22tf.Reshape\22(%1#0, %arg3) {_xla_outside_compilation = \220\22} : (tensor<1x1120x?xi32>, tensor<2xi32>) -> tensor<1120x?xi32> loc(#loc11)\0A    %3 = \22tf.Shape\22(%2) {_xla_outside_compilation = \220\22} : (tensor<1120x?xi32>) -> tensor<2xi32> loc(#loc12)\0A    return %1#1, %1#2, %2, %3 : tensor<1x1120x?xi32>, tensor<1x1120x?xi32>, tensor<1120x?xi32>, tensor<2xi32> loc(#loc9)\0A  } loc(#loc9)\0A} loc(#loc)\0A#loc = loc(unknown)\0A#loc4 = loc(\22Split:\22)\0A#loc5 = loc(\22split\22)\0A#loc6 = loc(\22Reshape_5\22)\0A#loc7 = loc(\22Shape:\22)\0A#loc8 = loc(\22Shape_4\22)\0A#loc10 = loc(fused[#loc4, #loc5])\0A#loc11 = loc(fused[#loc1, #loc6])\0A#loc12 = loc(fused[#loc7, #loc8])\0A", recv_key = "host_compute_channel_0_retvals", send_key = "host_compute_channel_0_args"}> : (tensor<3360x8xi32>, tensor<3xi32>, tensor<i32>, tensor<2xi32>) -> (tensor<1x1120x?xi32>, tensor<1x1120x?xi32>, tensor<1120x?xi32>, tensor<2xi32>)
    return
    }
}