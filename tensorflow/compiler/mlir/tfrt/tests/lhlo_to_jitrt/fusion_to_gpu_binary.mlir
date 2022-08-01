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

// RUN: lhlo-tfrt-opt %s --split-input-file \
// RUN:   -lmhlo-to-gpu-binary \
// RUN: | FileCheck %s

// CHECK: module attributes {gpu.container_module} {
// CHECK: gpu.module @[[gpu_module:.*]] attributes {
// CHECK-SAME: binary = "
// CHECK-SAME:   .visible .entry _fusion(
// CHECK-SAME:     .param .u64 _fusion_param_0
// CHECK-SAME:   )
// CHECK-SAME: "} {
// CHECK: gpu.func @[[kernel:.*]](%arg0: memref<4096xf32>) kernel {
// CHECK:  gpu.return
// CHECK: }
// CHECK: func @fusion(%arg0: memref<4096xf32>) {
func.func @fusion(%arg0: memref<4096xf32>) {

    // CHECK-DAG: %[[bx:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[by:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[bz:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[tx:.*]] = arith.constant 256 : index
    // CHECK-DAG: %[[ty:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[tz:.*]] = arith.constant 1 : index
    // CHECK: gpu.launch_func @[[gpu_module]]::@[[kernel]]
    // CHECK-SAME: blocks in (%[[bx]], %[[by]], %[[bz]])
    // CHECK-SAME: threads in (%[[tx]], %[[ty]], %[[tz]])
    // CHECK-SAME: args(%arg0 : memref<4096xf32>)
    "lmhlo.fusion"() ({
      %tensor = bufferization.to_tensor %arg0 : memref<4096xf32>
      %result = mhlo.add %tensor, %tensor : tensor<4096xf32>
      memref.tensor_store %result, %arg0 : memref<4096xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()

  "lmhlo.terminator"() : () -> ()
}

// -----

memref.global "private" constant @zero : memref<f32> = dense<0x80000000>
memref.global "private" constant @ones : memref<8xf32> = dense<
  "0x3F8000003F8000003F8000003F8000003F8000003F8000003F8000003F800000"
>


// CHECK: module attributes {gpu.container_module} {
// CHECK: gpu.module @[[gpu_module:.*]] attributes {
// CHECK-SAME: binary = "
// CHECK-SAME:   .visible .global .align 128 .b8 zero[4] = {0, 0, 0, 128};
// CHECK-SAME:   .visible .global .align 128 .b8 ones[32];
// CHECK-SAME:   .visible .entry _fusion(
// CHECK-SAME:     .param .u64 _fusion_param_0,
// CHECK-SAME:     .param .u64 _fusion_param_1
// CHECK-SAME:   )
// CHECK-SAME: ", constants = [@ones]} {
// CHECK: gpu.func @[[kernel:.*]](
// CHECK-SAME:   %arg0: memref<8x128xf32>, %arg1: memref<8xf32>
// CHECK-SAME: ) kernel {
// CHECK:   gpu.return
// CHECK: }
// CHECK: func @fusion(%arg0: memref<8x128xf32>, %arg1: memref<8xf32>) {
func.func @fusion(%arg0: memref<8x128xf32>, %arg1: memref<8xf32>) {

    %zero = memref.get_global @zero : memref<f32>
    %ones = memref.get_global @ones : memref<8xf32>

    // CHECK-DAG: %[[bx:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[by:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[bz:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[tx:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[ty:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[tz:.*]] = arith.constant 1 : index
    // CHECK: gpu.launch_func @[[gpu_module]]::@[[kernel]]
    // CHECK-SAME: blocks in (%[[bx]], %[[by]], %[[bz]])
    // CHECK-SAME: threads in (%[[tx]], %[[ty]], %[[tz]])
    // CHECK-SAME: args(%arg0 : memref<8x128xf32>, %arg1 : memref<8xf32>)
    "lmhlo.fusion"() ({
      %clamp = bufferization.to_tensor %zero : memref<f32>
      %bias = bufferization.to_tensor %ones : memref<8xf32>
      %tensor = bufferization.to_tensor %arg0 : memref<8x128xf32>
      %0 = "mhlo.reduce"(%tensor, %clamp) ({
      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %max = mhlo.maximum %arg2, %arg3 : tensor<f32>
        "mhlo.return"(%max) : (tensor<f32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>}
        : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      %1 = mhlo.add %0, %bias : tensor<8xf32>
      memref.tensor_store %1, %arg1 : memref<8xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()

  "lmhlo.terminator"() : () -> ()
}
