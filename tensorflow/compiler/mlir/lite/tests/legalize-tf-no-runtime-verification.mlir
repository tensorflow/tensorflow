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
// RUN: litert-opt %s -tfl-prepare-tf -tfl-legalize-tf='run-tfl-runtime-verification=false'  -tfl-optimize | FileCheck %s

func.func @broadcast_to_bf16(%arg0: tensor<3xbf16>, %arg1: tensor<2xi64>) -> tensor<3x3xbf16> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xbf16>, tensor<2xi64>) -> tensor<3x3xbf16>
  func.return %0: tensor<3x3xbf16>

// CHECK-LABEL: broadcast_to_bf16
// CHECK:  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xbf16>, tensor<2xi64>) -> tensor<3x3xbf16>
// CHECK:  return %0 : tensor<3x3xbf16>
}
