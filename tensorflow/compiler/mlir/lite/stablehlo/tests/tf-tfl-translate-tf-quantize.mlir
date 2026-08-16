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

//RUN: tf_tfl_translate --post-training-quantization --enable-stablehlo-conversion --input-mlir --output-mlir %s -o - | FileCheck %s


module {
func.func @tfInplaceUpdate(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
  %1 = arith.constant dense<1> : tensor<1xi32>
  %2 = arith.constant dense<2.0> : tensor<1x1x2xf32>
  %3 = "tf.InplaceUpdate"(%arg0, %1, %2) {device = ""}
    : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
  func.return %3 : tensor<2x1x2xf32>
}
}

//CHECK: module {
//CHECK-NEXT:  func.func @main(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
//CHECK-DAG:    %[[c0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<1x1x2xf32>
//CHECK-DAG:    %[[c1:.+]] = stablehlo.constant dense<1> : tensor<i32>
//CHECK-DAG:    %[[c2:.+]] = stablehlo.constant dense<0> : tensor<i32>
//CHECK-NEXT:    %[[c3:.+]] = stablehlo.dynamic_update_slice %arg0, %[[c0]], %[[c1]], %[[c2]], %[[c2]] : (tensor<2x1x2xf32>, tensor<1x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x2xf32>
//CHECK-NEXT:    return %[[c3:.+]] : tensor<2x1x2xf32>
//CHECK-NEXT:  }
//CHECK-NEXT:}
