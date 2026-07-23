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
// RUN: tf-reduce %s -reduction-tree='traversal-mode=0 test=%S/reducer/unsupported-op-test.sh' | FileCheck %s

// CHECK: @target_function
func.func @target_function() -> tensor<i32> {
  %0 = "tf_device.cluster"() ({
    // CHECK: tf.UnsupportedOp
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: tf.Identity
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    // CHECK-NOT: tf.Identity
    %3 = "tf.Identity"(%2) : (tensor<i32>) -> tensor<i32>
    %4 = "tf.Identity"(%3) : (tensor<i32>) -> tensor<i32>
    // CHECK: tf_device.return
    tf_device.return %4 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}
