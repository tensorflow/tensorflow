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
// RUN: tf-opt -pass-pipeline='builtin.module(func.func(infeed-ops-xla-adjust-layout))' %s | env FILECHECK_OPTS="" FileCheck %s

func.func @infeed_dequeue_tuple() -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>) {
  // CHECK: [[TOKEN:%.*]] = mhlo.create_token : !mhlo.token
  %0 = "mhlo.create_token"() : () -> !mhlo.token

  // CHECK:               [[INFEED:%.*]]:3 = "mhlo.infeed"([[TOKEN]]) <{
  // CHECK-SAME{LITERAL}:   infeed_config = "", layout = [[1, 3, 2, 0], [1, 2, 0]]
  // CHECK-SAME:          }> : (!mhlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !mhlo.token)
  %1:3 = "mhlo.infeed"(%0) {infeed_config = ""} : (!mhlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !mhlo.token)

  // CHECK: return [[INFEED]]#0, [[INFEED]]#1
  func.return %1#0, %1#1 : tensor<1x8x4x4xi32>, tensor<1x100x1xf32>
}
