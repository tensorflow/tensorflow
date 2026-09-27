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
// RUN: tf-tfrt-opt %s -inline | FileCheck %s

func.func @_tfrt_fallback_init(%arg0: !tfrt.chain) -> !tfrt.chain {
  %0 = tfrt_fallback_async.createop(%arg0) key(0) device("/device:CPU:0") "tf.Less"() {T = i32} num_args(2)
  tfrt.return %0 : !tfrt.chain
}

func.func @callee(%ch: !tfrt.chain, %arg: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor) {
  %const = tfrt_fallback_async.const_dense_tensor dense<9> : tensor<i32> {_tfrt_cost = 1 : i64}
  %result = tfrt_fallback_async.executeop key(0) cost(3) device("/device:CPU:0") "tf.Less"(%arg, %const) {T = i32} : 1
  tfrt.return %ch, %result : !tfrt.chain, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_inline_fallback_ops
// CHECK-SAME: ([[ch:%.*]]: !tfrt.chain, [[arg:%.*]]: !tfrt_fallback.tf_tensor
func.func @test_inline_fallback_ops(%ch: !tfrt.chain, %arg: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor) {
  // CHECK-NOT: tfrt.call
  // CHECK: [[const:%.*]] = tfrt_fallback_async.const_dense_tensor dense<9> : tensor<i32>
  // CHECK-NEXT: [[result:%.*]] = tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/device:CPU:0") "tf.Less"([[arg]], [[const]]) {T = i32} : 1
  // CHECK-NEXT: tfrt.return [[ch]], [[result]] : !tfrt.chain, !tfrt_fallback.tf_tensor
  %out_ch, %result = tfrt.call @callee(%ch, %arg) : (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  tfrt.return %out_ch, %result : !tfrt.chain, !tfrt_fallback.tf_tensor
}
