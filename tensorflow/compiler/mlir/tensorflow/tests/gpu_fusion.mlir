// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s -tf-gpu-op-fusion | FileCheck %s

// Test the op-fusion pass specific to the GPU target.

// CHECK-LABEL: func @FusedBatchNormRelu
func.func @FusedBatchNormRelu(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}_FusedBatchNormEx
// CHECK-NEXT: return %[[Y]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %relu = "tf.Relu"(%y#0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @FusedBatchNormAddRelu
func.func @FusedBatchNormAddRelu(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}_FusedBatchNormEx
// CHECK-NEXT: return %[[Y]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %add = "tf.AddV2"(%arg0, %y#0) : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = "tf.Relu"(%add) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @FusedBatchNormAddReluTwoUses
func.func @FusedBatchNormAddReluTwoUses(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) {
// Since the tf.AddV2 op has two uses, we have a _FusedBatchNormEx without the
// Relu activation and we only fuse the add.
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}_FusedBatchNormEx
// CHECK-NEXT: %[[relu:[a-z0-9]*]] ={{.*}}Relu"(%[[Y]]
// CHECK-NEXT: return %[[relu]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %add = "tf.AddV2"(%arg0, %y#0) : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = "tf.Relu"(%add) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu, %add  : tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @TrainingFusedBatchNormRelu
func.func @TrainingFusedBatchNormRelu(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // We don't fuse in training right now
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}FusedBatchNorm
// CHECK-NEXT: %[[relu:[a-z0-9]*]] ={{.*}}Relu"(%[[Y]]
// CHECK-NEXT: return %[[relu]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %relu = "tf.Relu"(%y#0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu : tensor<8x8x8x8xf32>
}

