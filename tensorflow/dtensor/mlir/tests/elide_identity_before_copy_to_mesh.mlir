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
// RUN: dtensor-opt %s -split-input-file -dtensor-elide-identity-before-copy-to-mesh | FileCheck %s

// Check that identity before CopyToMeshGrad is elided.
// CHECK-LABEL: func @check_elide_identity
func.func @check_elide_identity() -> (tensor<4xi32>) {
    // CHECK: %[[CONST:.*]] = "tf.Const"()
    // CHECK-NEXT: %[[CONST_1:.*]] = "tf.Const"()
    // CHECK-NEXT: "tf.CopyToMeshGrad"(%[[CONST]], %[[CONST_1]])

    %cst = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
    %cst_1 = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = "tf.Identity"(%cst) : (tensor<4xi32>) -> tensor<4xi32>
    %2 = "tf.CopyToMeshGrad"(%1, %cst_1) {reference_layout=""}: (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    func.return %2 : tensor<4xi32>
}


