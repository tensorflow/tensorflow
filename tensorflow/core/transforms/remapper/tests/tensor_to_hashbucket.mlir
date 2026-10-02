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
// RUN: tfg-transforms-opt --tfg-remapper %s | FileCheck %s

// -----

// CHECK-LABEL: tfg.func @tensor_to_hashbucket_test
tfg.func @tensor_to_hashbucket_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = i8, shape = #tf_type.shape<8x32x32x3>} : () -> (tensor<*xi8>)
  // CHECK: %[[ASSTRING:.*]], {{.*}} name("to_string")
  %AsString, %ctl_0 = AsString(%Placeholder) device("/device:CPU:0") name("to_string") {T = i8, fill = "", precision = -1 : i64, scientific = false, shortest = false, width = -1 : i64} : (tensor<*xi8>) -> (tensor<*x!tf_type.string>)
  // CHECK: _TensorToHashBucketFast(%[[PLACEHOLDER:.*]]) {{.*}} name("to_bucket") {T = {{.*}}, num_buckets = {{.*}}}
  %StringToHashBucketFast, %ctl_1 = StringToHashBucketFast(%AsString) device("/device:CPU:0") name("to_bucket") {num_buckets = 100 : i64} : (tensor<*x!tf_type.string>) -> (tensor<*xi64>)
  return
}
