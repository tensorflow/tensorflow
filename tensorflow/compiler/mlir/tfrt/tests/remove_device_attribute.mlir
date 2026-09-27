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
// RUN: tf-tfrt-opt -tfrt-remove-device-attribute %s | FileCheck %s

func.func @test(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %0 = corert.get_op_handler %arg0 "cpu"
  // CHECK: %[[RESULT:.*]] = corert.executeop(%[[ARG_0:.*]]) "tf.MatMul"(%[[ARG_1:.*]], %[[ARG_1]]) {T = f32, transpose_a = false, transpose_b = false} : 1
  %1 = corert.executeop(%0) "tf.MatMul"(%arg1, %arg1) {T = f32, device = "cpu", transpose_a = false, transpose_b = false} : 1
  tfrt.return %arg0, %1 : !tfrt.chain, !corert.tensorhandle
}
