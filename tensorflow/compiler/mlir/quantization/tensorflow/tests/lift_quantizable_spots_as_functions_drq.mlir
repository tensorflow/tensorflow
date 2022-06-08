// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions-drq | FileCheck %s

// CHECK-LABEL: float_matmul
func.func @float_matmul(%arg0: tensor<1x12x12x512xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
  %out = "tf.MatMul"(%arg0, %cst) {
    device = "", transpose_a = false, transpose_b = false
  } : (tensor<1x12x12x512xf32>, tensor<512x512xf32>) -> tensor<*xf32>
  func.return %out : tensor<*xf32>

// CHECK: %[[CONST:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
// CHECK: %[[PARTITIONEDCALL:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST]])
// CHECK-SAME: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: f = @composite_matmul_fn_1}
// CHECK: return %[[PARTITIONEDCALL]]
// CHECK: }

// CHECK-LABEL: private @composite_matmul_fn_1
// CHECK-NEXT: %[[OUT:.*]] = "tf.MatMul"(%arg0, %arg1)
// CHECK-NEXT: return %[[OUT]]
}
