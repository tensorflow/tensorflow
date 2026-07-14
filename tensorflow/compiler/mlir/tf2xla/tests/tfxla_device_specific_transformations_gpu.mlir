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
// RUN: tf-opt "--tfxla-device-specific-transforms=device-type=XLA_GPU_JIT" -verify-diagnostics -split-input-file %s | FileCheck -dump-input=fail %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1399 : i32}} {

// CHECK-LABEL: stateless_op
func.func @stateless_op() -> tensor<i32> {
  // CHECK: %cst = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.StatelessRandomGetAlg"() {device = ""} : () -> tensor<i32>
  return %0 : tensor<i32>
}

}
